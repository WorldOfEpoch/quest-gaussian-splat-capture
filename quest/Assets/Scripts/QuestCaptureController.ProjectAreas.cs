using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace QuestSplatBridge
{
    public sealed partial class QuestCaptureController
    {
        private readonly Dictionary<string, string> projectAreaLabels = new Dictionary<string, string>();
        private readonly HashSet<string> projectsInArea = new HashSet<string>();
        private readonly Dictionary<string, Vector3> projectAreaPositions = new Dictionary<string, Vector3>();
        private readonly Dictionary<string, float> projectAreaCheckedAt = new Dictionary<string, float>();
        private bool projectAreaSearchRunning;
        private int projectAreaLocationEpoch;
        private const float ProjectNearbyDistanceMeters = 8f;
        private const float ProjectAreaResultLifetimeSeconds = 15f;

        // These are local spatial-anchor checks. The headset boundary is not a room identity.
        // Keep the native request owned until it finishes, even after its menu closes.
        private IEnumerator LocateProjectAreas(int generation)
        {
            while (projectAreaSearchRunning && ProjectAreaGenerationCurrent(generation))
                yield return null;
            if (!ProjectAreaGenerationCurrent(generation)) yield break;
            projectAreaSearchRunning = true;
            var operation = LocateProjectAreasAsync(generation, projectAreaLocationEpoch);
            while (!operation.IsCompleted) yield return null;
            if (operation.IsFaulted) Debug.LogException(operation.Exception);
            if (ProjectAreaGenerationCurrent(generation) && projectsScreen == ProjectsScreen.List)
            { OrderProjectsByLocation(); ShowProjectChoice(); }
        }

        private async Task LocateProjectAreasAsync(int generation, int locationEpoch)
        {
            var probes = new List<GameObject>();
            var startedAt = Time.realtimeSinceStartup;
            try
            {
                InvalidateProjectAreaResults();
                // Invalidation starts this operation's own tracking epoch.
                locationEpoch = projectAreaLocationEpoch;
                if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
                {
                    projectMenuStatus = "Allow Spatial Data to locate saved scans\nY Close; reopen Projects after allowing";
                    OVRPermissionsRequester.Request(new[] { OVRPermissionsRequester.Permission.Scene });
                    return;
                }
                projectAreaLabels.Clear();
                projectsInArea.Clear();
                projectAreaPositions.Clear();
                projectAreaCheckedAt.Clear();
                var byUuid = new Dictionary<Guid, List<string>>();
                var candidates = projectsScreen == ProjectsScreen.ConfirmSwitch && proposedProfile != null
                    ? new[] { new ProjectInfo { id = proposedProfile.project_id, name = proposedProfile.name } }
                    : visibleProjects.ToArray();
                foreach (var project in candidates)
                {
                    var profile = projectsScreen == ProjectsScreen.ConfirmSwitch ? proposedProfile :
                        (localProfiles.TryGetValue(project.id, out var local) ? local : null);
                    if (profile == null || !ValidProfile(profile))
                    { projectAreaLabels[project.id] = "No local scan"; continue; }
                    if (!TryReadProjectAnchor(profile, out var uuid))
                    {
                        var recordExists = File.Exists(Path.Combine(Application.persistentDataPath,
                            "anchor-" + profile.session_id + ".json"));
                        projectAreaLabels[project.id] = !profile.has_capture && !recordExists
                            ? "Ready for first scan" : "No local scan";
                        continue;
                    }
                    projectAreaLabels[project.id] = "Checking location";
                    if (sessionAnchor != null && sessionAnchor.Uuid == uuid)
                    {
                        if (AnchorReady && ProjectAreaTrackingAvailable())
                            RecordProjectArea(project.id, sessionAnchor.transform.position);
                        else projectAreaLabels[project.id] = "Not located here";
                        continue;
                    }
                    if (!byUuid.TryGetValue(uuid, out var ids))
                    { ids = new List<string>(); byUuid.Add(uuid, ids); }
                    ids.Add(project.id);
                }

                var uuids = byUuid.Keys.ToArray();
                for (var offset = 0; offset < uuids.Length; offset += 50)
                {
                    if (!ProjectAreaSearchCurrent(generation, locationEpoch, startedAt)) break;
                    var loaded = new List<OVRSpatialAnchor.UnboundAnchor>();
                    var batch = uuids.Skip(offset).Take(50).ToArray();
                    var result = await OVRSpatialAnchor.LoadUnboundAnchorsAsync(batch, loaded);
                    if (!ProjectAreaSearchCurrent(generation, locationEpoch, startedAt)) break;
                    if (!result.Success) continue;
                    // Localization is bounded to four simultaneous requests and a finite SDK timeout.
                    for (var index = 0; index < loaded.Count; index += 4)
                    {
                        if (!ProjectAreaSearchCurrent(generation, locationEpoch, startedAt)) break;
                        var operations = new List<Task>();
                        for (var item = index; item < Math.Min(index + 4, loaded.Count); item++)
                        {
                            var unbound = loaded[item];
                            if (byUuid.TryGetValue(unbound.Uuid, out var ids))
                                operations.Add(LocateProjectAreaAsync(unbound, ids, probes,
                                    generation, locationEpoch, startedAt));
                        }
                        await Task.WhenAll(operations);
                    }
                }
            }
            catch (Exception error) { Debug.LogException(error); }
            finally
            {
                // Destroy releases runtime handles. Never erase the persisted spatial anchor.
                foreach (var probe in probes) if (probe != null) Destroy(probe);
                // Destroy is deferred until the end of the Unity frame. A successor must
                // not fetch one of these UUIDs while it remains bound in the SDK registry.
                if (probes.Count != 0) await Task.Delay(100);
                projectAreaSearchRunning = false;
                if (ProjectAreaGenerationCurrent(generation))
                {
                    foreach (var id in projectAreaLabels.Keys.ToArray())
                        if (projectAreaLabels[id] == "Checking location")
                            projectAreaLabels[id] = "Not located here";
                    if (locationEpoch != projectAreaLocationEpoch || !ProjectAreaTrackingAvailable())
                        InvalidateProjectAreaResults();
                }
            }
        }

        private async Task LocateProjectAreaAsync(OVRSpatialAnchor.UnboundAnchor unbound,
            List<string> projectIds, List<GameObject> probes, int generation, int locationEpoch, float startedAt)
        {
            try
            {
                if (!unbound.Localized && !await unbound.LocalizeAsync(6)) return;
                if (!ProjectAreaSearchCurrent(generation, locationEpoch, startedAt) ||
                    !unbound.TryGetPose(out var pose)) return;
                var probe = new GameObject("Project location check");
                probes.Add(probe);
                probe.transform.SetPositionAndRotation(pose.position, pose.rotation);
                var anchor = probe.AddComponent<OVRSpatialAnchor>();
                unbound.BindTo(anchor);
                var readyBy = Time.realtimeSinceStartup + 4f;
                var stableSince = -1f;
                while (ProjectAreaSearchCurrent(generation, locationEpoch, startedAt) &&
                       anchor != null && Time.realtimeSinceStartup < readyBy)
                {
                    if (anchor.IsTracked && ProjectAreaTrackingAvailable())
                    {
                        if (stableSince < 0) stableSince = Time.realtimeSinceStartup;
                        if (Time.realtimeSinceStartup - stableSince >= 0.75f)
                        {
                            foreach (var id in projectIds) RecordProjectArea(id, anchor.transform.position);
                            return;
                        }
                    }
                    else stableSince = -1f;
                    await Task.Delay(100);
                }
            }
            catch (Exception error) { Debug.LogException(error); }
        }

        private bool TryReadProjectAnchor(ProjectProfile profile, out Guid uuid)
        {
            uuid = Guid.Empty;
            if (!ValidProfile(profile)) return false;
            var path = Path.Combine(Application.persistentDataPath, "anchor-" + profile.session_id + ".json");
            if (!File.Exists(path)) return false;
            try
            {
                var record = JsonUtility.FromJson<AnchorRecord>(File.ReadAllText(path));
                return record != null && record.session_id == profile.session_id &&
                       record.origin_id == profile.tracking_origin_id &&
                       Guid.TryParse(record.anchor_id, out uuid) && uuid != Guid.Empty;
            }
            catch (Exception error) { Debug.LogException(error); return false; }
        }

        private void RecordProjectArea(string projectId, Vector3 position)
        {
            if (!ProjectAreaTrackingAvailable()) return;
            var distance = Vector3.Distance(statusCamera.transform.position, position);
            if (!Finite(distance) || distance > ProjectNearbyDistanceMeters)
            { projectAreaLabels[projectId] = "Not located here"; return; }
            // A valid position must also be finite; NaN distances otherwise pass a greater-than test.
            if (!Finite(position.x) || !Finite(position.y) || !Finite(position.z)) return;
            projectAreaPositions[projectId] = position;
            projectAreaCheckedAt[projectId] = Time.realtimeSinceStartup;
            projectsInArea.Add(projectId);
            projectAreaLabels[projectId] = "Aligned nearby";
        }

        private bool ProjectAreaGenerationCurrent(int generation) =>
            this && generation == projectMenuGeneration &&
            (projectsScreen == ProjectsScreen.List || projectsScreen == ProjectsScreen.ConfirmSwitch);

        private bool ProjectAreaSearchCurrent(int generation, int locationEpoch, float startedAt) =>
            ProjectAreaGenerationCurrent(generation) && locationEpoch == projectAreaLocationEpoch &&
            Time.realtimeSinceStartup - startedAt < 45f && ProjectAreaTrackingAvailable();

        private bool ProjectAreaTrackingAvailable() =>
            statusCamera != null && OVRManager.hasVrFocus && OVRManager.isHmdPresent;

        // Call before treating a cached list result as selectable. Confirmation must localize again.
        private bool ProjectAreaResultIsCurrent(string projectId) =>
            OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene) &&
            projectsInArea.Contains(projectId) && ProjectAreaTrackingAvailable() &&
            projectAreaCheckedAt.TryGetValue(projectId, out var checkedAt) &&
            Time.realtimeSinceStartup - checkedAt < ProjectAreaResultLifetimeSeconds &&
            projectAreaPositions.TryGetValue(projectId, out var position) &&
            Vector3.Distance(statusCamera.transform.position, position) <= ProjectNearbyDistanceMeters;

        // The existing recenter/focus owners call this; a probe result cannot survive a new tracking frame.
        private void InvalidateProjectAreaResults()
        {
            ++projectAreaLocationEpoch;
            foreach (var id in projectsInArea) projectAreaLabels[id] = "Not located here";
            projectsInArea.Clear();
            projectAreaPositions.Clear();
            projectAreaCheckedAt.Clear();
            if (libraryOpen && projectsScreen == ProjectsScreen.List) ShowProjectChoice();
        }
    }
}
