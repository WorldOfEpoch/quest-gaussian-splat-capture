using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

namespace QuestSplatBridge
{
    public sealed partial class QuestCaptureController
    {
        private OVRSpatialAnchor sessionAnchor;
        private bool anchorSaved;
        private bool anchorReadyAnnounced;
        private float anchorStableSince = -1f;
        private int anchorInitializationGeneration = -1;
        private bool AnchorReady => anchorSaved && sessionAnchor != null && sessionAnchor.IsTracked &&
            anchorStableSince >= 0 && Time.unscaledTime - anchorStableSince >= 0.75f;

        // The persisted anchor owns the session frame. Recenter and headset removal
        // only suspend capture while it relocalizes; they do not relabel old poses.
        private async void InitializeSessionAnchor()
        {
            if (!Configured() || anchorSaved || anchorInitializationGeneration == configGeneration) return;
            if (!metaPlatformReady) { status = metaPlatformStatus; return; }
            var generation = configGeneration;
            anchorInitializationGeneration = generation;
            var sessionId = config.session_id;
            var originId = config.tracking_origin_id;
            var path = Path.Combine(Application.persistentDataPath, "anchor-" + sessionId + ".json");
            try
            {
                if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
                {
                    status = "Allow Spatial Data to save scan location";
                    for (var i = 0; i < 300 && AnchorGenerationCurrent(generation, sessionId) &&
                        !OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene); i++)
                        await Task.Delay(100);
                    if (!AnchorGenerationCurrent(generation, sessionId)) return;
                    if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
                        throw new InvalidOperationException("Spatial Data permission required; allow it, then A retries");
                }
                if (File.Exists(path))
                {
                    var record = JsonUtility.FromJson<AnchorRecord>(File.ReadAllText(path));
                    if (record.session_id != sessionId || record.origin_id != originId ||
                        !Guid.TryParse(record.anchor_id, out var uuid))
                        throw new InvalidDataException("Saved scan anchor does not match session");
                    status = "Restoring scan location";
                    var loaded = new List<OVRSpatialAnchor.UnboundAnchor>();
                    var result = await OVRSpatialAnchor.LoadUnboundAnchorsAsync(new[] { uuid }, loaded);
                    if (!AnchorGenerationCurrent(generation, sessionId)) return;
                    if (!result.Success || loaded.Count != 1 || !await loaded[0].LocalizeAsync(12))
                        throw new InvalidOperationException("Scan anchor not found; return to the scanned area and restart app");
                    if (!AnchorGenerationCurrent(generation, sessionId)) return;
                    sessionAnchor = new GameObject("Persisted scan origin").AddComponent<OVRSpatialAnchor>();
                    loaded[0].BindTo(sessionAnchor);
                }
                else
                {
                    if (sessionHasCapture)
                        throw new InvalidOperationException("Legacy scan needs a new anchored session");
                    status = "Saving scan location";
                    var origin = new GameObject("Persisted scan origin");
                    origin.transform.SetPositionAndRotation(statusCamera.transform.position, Quaternion.identity);
                    sessionAnchor = origin.AddComponent<OVRSpatialAnchor>();
                    for (var i = 0; i < 150 && AnchorGenerationCurrent(generation, sessionId) &&
                         sessionAnchor != null && !sessionAnchor.Localized; i++)
                        await Task.Delay(100);
                    if (!AnchorGenerationCurrent(generation, sessionId)) return;
                    if (sessionAnchor == null || !sessionAnchor.Localized)
                        throw new InvalidOperationException("Unable to locate scan anchor; look around and restart app");
                    var saved = await sessionAnchor.SaveAnchorAsync();
                    if (!AnchorGenerationCurrent(generation, sessionId)) return;
                    if (!saved.Success) throw new InvalidOperationException("Could not save scan location: " + saved.Status);
                    var record = new AnchorRecord { session_id = sessionId, origin_id = originId,
                        anchor_id = sessionAnchor.Uuid.ToString() };
                    File.WriteAllText(path + ".tmp", JsonUtility.ToJson(record));
                    File.Move(path + ".tmp", path);
                }
                sessionOrigin = sessionAnchor.transform;
                anchorSaved = true;
                originInvalidated = false;
                anchorStableSince = -1f;
                Debug.Log("Quest Splat: persistent scan anchor ready for " + sessionId);
            }
            catch (Exception error)
            {
                Debug.LogException(error);
                if (AnchorGenerationCurrent(generation, sessionId))
                {
                    if (!anchorSaved && sessionAnchor != null) Destroy(sessionAnchor.gameObject);
                    sessionAnchor = null;
                    originInvalidated = true; status = error.Message;
                }
            }
            finally { if (anchorInitializationGeneration == generation) anchorInitializationGeneration = -1; }
        }

        private bool AnchorGenerationCurrent(int generation, string sessionId) =>
            this && generation == configGeneration && config != null && config.session_id == sessionId;

        private void UpdateAnchorTracking()
        {
            if (!anchorSaved || sessionAnchor == null) return;
            if (!sessionAnchor.IsTracked || !OVRManager.hasVrFocus || !OVRManager.isHmdPresent)
            {
                anchorStableSince = -1f;
                anchorReadyAnnounced = false;
                capturePending = false;
                if (previewObject != null && previewAligned) previewObject.SetActive(false);
                UpdateCoverageTint();
                return;
            }
            if (anchorStableSince < 0)
            {
                anchorStableSince = Time.unscaledTime;
                status = "Scan located; hold briefly";
            }
            if (AnchorReady && !anchorReadyAnnounced)
            {
                anchorReadyAnnounced = true;
                if (!IsLibraryOpen) status = "Scan located; A captures";
            }
            if (AnchorReady && previewObject != null && previewAligned)
                previewObject.SetActive(IsPreviewSelected);
            UpdateCoverageTint();
        }

        [Serializable] private sealed class AnchorRecord
        {
            public string session_id, origin_id, anchor_id;
        }
    }
}
