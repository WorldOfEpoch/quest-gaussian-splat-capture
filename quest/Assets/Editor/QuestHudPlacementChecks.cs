using System;
using System.Reflection;
using System.Linq;
using UnityEngine;

namespace QuestSplatBridge.Editor
{
    // Exercise the production controller with real Unity cameras/transforms.
    // No headset, HTTP, capture or project creation is involved in this fixture.
    public static class QuestHudPlacementChecks
    {
        private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        private static readonly Type Controller = typeof(QuestCaptureController);

        public static void Run()
        {
            var root = new GameObject("HUD placement regression fixture");
            try
            {
                var cameraObject = new GameObject("Fixture camera");
                cameraObject.transform.SetParent(root.transform);
                var camera = cameraObject.AddComponent<Camera>();
                camera.transform.SetPositionAndRotation(new Vector3(2, 1.6f, 3),
                    Quaternion.Euler(10, 65, 17));
                var panel = new GameObject("Fixture panel").transform;
                panel.SetParent(root.transform);
                var capture = root.AddComponent<QuestCaptureController>();
                capture.enabled = false;
                Set(capture, "statusCamera", camera);
                Set(capture, "legendPanel", panel);
                Call(capture, "EnsureFullMenuPlacement");
                var center = camera.WorldToViewportPoint(panel.TransformPoint(new Vector3(0, -0.20f, 0)));
                Require(Mathf.Abs(center.x - 0.5f) < 0.0001f && Mathf.Abs(center.y - 0.5f) < 0.0001f &&
                    center.z > 0.85f, "First menu must be centered in front of the wearer");

                camera.transform.SetPositionAndRotation(new Vector3(4, 1.7f, -2), Quaternion.Euler(0, -20, 0));
                Call(capture, "CenterFullMenu");
                center = camera.WorldToViewportPoint(panel.TransformPoint(new Vector3(0, -0.20f, 0)));
                Require(Mathf.Abs(center.x - 0.5f) < 0.0001f && Mathf.Abs(center.y - 0.5f) < 0.0001f,
                    "Unplaced menu must remain centered when the headset is picked up");

                var firstPosition = panel.position;
                var firstRotation = panel.rotation;
                var handStart = new Vector3(2.2f, 1, 3.1f);
                Call(capture, "BeginMenuGrab", handStart);
                Require(Vector3.Distance(panel.position, firstPosition) < 0.0001f,
                    "Grip pickup must not teleport the menu");
                var translation = new Vector3(-0.45f, 0.1f, 0.15f);
                Call(capture, "MoveMenuGrab", handStart + translation);
                var placed = firstPosition + translation;
                Require(Vector3.Distance(panel.position, placed) < 0.0001f &&
                    Quaternion.Angle(panel.rotation, firstRotation) < 0.001f,
                    "Grip must translate the panel without rotating it");

                var mode = Controller.GetNestedType("HudDisplayMode", BindingFlags.NonPublic);
                foreach (var next in new[] { "Compact", "Hidden", "Full", "Hidden", "Full" })
                {
                    Call(capture, "SetHudMode", Enum.Parse(mode, next));
                    camera.transform.SetPositionAndRotation(new Vector3(-4, 2, 7), Quaternion.Euler(0, -75, 0));
                    Call(capture, "UpdateHudPlacement");
                    Require(panel.gameObject.activeSelf == (next != "Hidden"), "HUD visibility is wrong");
                    if (next == "Full")
                        Require(Vector3.Distance(panel.position, placed) < 0.0001f &&
                            Quaternion.Angle(panel.rotation, firstRotation) < 0.001f,
                            "Full menu must restore its world placement after head movement and mode changes");
                }

                Set(capture, "libraryOpen", true);
                Set(capture, "projectName", "Fixture original room");
                Call(capture, "RefreshLibraryChoices");
                Require(((string)Get(capture, "status")).Contains("Projects / New room"),
                    "New-room selector must remain the first library choice");
                var screen = Controller.GetNestedType("ProjectsScreen", BindingFlags.NonPublic);
                Set(capture, "projectsScreen", Enum.Parse(screen, "List"));
                Call(capture, "ShowProjectChoice");
                Call(capture, "UpdateHudPlacement");
                Require(((string)Get(capture, "projectMenuStatus")).Contains("New project") &&
                    Vector3.Distance(panel.position, placed) < 0.0001f,
                    "Project submenu must preserve its contents and the placed menu pose");
                var projectNotice = (string)Get(capture, "projectMenuStatus");
                Set(capture, "status", "PC upload received; unrelated background status");
                Require((string)Controller.GetProperty("CurrentMenuStatus", Private).GetValue(capture) == projectNotice,
                    "Background upload notices must not replace project choices");
                Call(capture, "HandleProjectMenuControl", 1);
                Require(((string)Get(capture, "projectMenuStatus")).Contains("Models in current project"),
                    "Models must remain available from the direct projects menu");
                var wrap = Controller.GetMethod("WrapHud", BindingFlags.Static | BindingFlags.NonPublic);
                var wrapped = (string)wrap.Invoke(null, new object[] { "PROJECTS\nNew project\n" + new string('x', 61), 28 });
                Require(wrapped.Split('\n').All(line => line.Length <= 28) && wrapped.StartsWith("PROJECTS\nNew project\n"),
                    "HUD wrapping must preserve explicit menu lines and bound long labels");
                Set(capture, "config", null);
                Require(!(bool)Controller.GetMethod("BridgePaired", Private).Invoke(capture, null),
                    "Unpaired headsets must not send project creation requests");
                var bridgeType = Controller.GetNestedType("BridgeConfig", BindingFlags.NonPublic);
                var bridge = Activator.CreateInstance(bridgeType);
                bridgeType.GetField("base_url").SetValue(bridge, "https://127.0.0.1:8766");
                bridgeType.GetField("token").SetValue(bridge, new string('t', 48));
                Set(capture, "config", bridge);
                Require((bool)Controller.GetMethod("BridgePaired", Private).Invoke(capture, null) &&
                    !(bool)Controller.GetMethod("Configured", Private).Invoke(capture, null),
                    "Valid PC pairing must allow recovery without an existing valid project/session");
                var profileType = Controller.GetNestedType("ProjectProfile", BindingFlags.NonPublic);
                var profile = Activator.CreateInstance(profileType);
                profileType.GetField("session_id").SetValue(profile, Guid.NewGuid().ToString());
                Require(!(bool)Controller.GetMethod("ProjectNeedsLocation", Private).Invoke(capture, new[] { profile }),
                    "A fresh project without an anchor must be allowed to start here");
                profileType.GetField("has_capture").SetValue(profile, true);
                Require((bool)Controller.GetMethod("ProjectNeedsLocation", Private).Invoke(capture, new[] { profile }) &&
                    !(bool)Controller.GetMethod("ProjectAreaResultIsCurrent", Private).Invoke(capture, new object[] { "unlocated" }),
                    "A captured session must not be eligible without a checked location");
                Set(capture, "libraryOpen", false);
                Set(capture, "projectsScreen", Enum.Parse(screen, "Models"));
                Call(capture, "SetHudMode", Enum.Parse(mode, "Full"));
                Require(Vector3.Distance(panel.position, placed) < 0.0001f,
                    "Returning from a submenu must not reset the menu position");
                CheckExistingProjectFirstScan(capture, bridge, profileType);
                // Explicit access recovers an off-screen panel without breaking a
                // currently visible placement or depending on a network request.
                Call(capture, "CenterFullMenu");
                var visiblePosition = panel.position;
                Call(capture, "ShowFullMenu");
                Require(Vector3.Distance(panel.position, visiblePosition) < 0.0001f,
                    "Opening a visible full menu must preserve its placement");
                camera.transform.position += camera.transform.forward * 4;
                Call(capture, "SetHudMode", Enum.Parse(mode, "Hidden"));
                Call(capture, "ShowFullMenu");
                center = camera.WorldToViewportPoint(panel.TransformPoint(new Vector3(0, -0.20f, 0)));
                Require(panel.gameObject.activeSelf && Mathf.Abs(center.x - 0.5f) < 0.0001f &&
                    Mathf.Abs(center.y - 0.5f) < 0.0001f && center.z > 0.4f,
                    "Explicit menu access must recover a hidden panel behind the wearer");
                Call(capture, "SetHudMode", Enum.Parse(mode, "Compact"));
                Call(capture, "ReportCaptureIssue", "Fixture capture blocked");
                Require(Get(capture, "hudMode").ToString() == "Full" && panel.gameObject.activeSelf,
                    "A capture error must not disappear behind compact counts");
                Set(capture, "requestedPreview", null);
                Call(capture, "SetHudMode", Enum.Parse(mode, "Compact"));
                Call(capture, "Preview", "splat");
                Require(Get(capture, "hudMode").ToString() == "Full" &&
                    ((string)Get(capture, "status")).Contains("Loading latest splat"),
                    "Requesting a missing splat must show progress rather than empty compact counts");
                CheckRoomOverlayPlacement(capture, camera, root.transform);
                Debug.Log("PASS: Quest HUD first-open centering, no-jump pickup, world translation, mode persistence, project menu contents and submenu placement");
            }
            finally { UnityEngine.Object.DestroyImmediate(root); }
        }

        private static void CheckRoomOverlayPlacement(QuestCaptureController capture, Camera camera, Transform root)
        {
            var anchor = new GameObject("Fixture persisted scan anchor").transform;
            anchor.SetParent(root);
            anchor.SetPositionAndRotation(new Vector3(4, 1.5f, -2), Quaternion.Euler(0, 61, 0));
            var scan = new GameObject("Fixture room overlay").transform;
            scan.SetParent(root);
            var expected = Matrix4x4.TRS(new Vector3(0.3f, -0.7f, 1.1f),
                Quaternion.Euler(90, 47, -6), Vector3.one * 1.43f);
            var matrix = new float[16];
            for (var r = 0; r < 4; r++) for (var c = 0; c < 4; c++) matrix[r * 4 + c] = expected[r, c];
            Set(capture, "sessionOrigin", anchor);
            var headPosition = camera.transform.position;
            var headRotation = camera.transform.rotation;
            Call(capture, "ApplyRoomPreviewTransform", scan, matrix);
            var localPosition = scan.localPosition;
            var localRotation = scan.localRotation;
            var localScale = scan.localScale;
            Require(scan.parent == anchor && Vector3.Distance(localPosition, expected.GetColumn(3)) < 0.0001f &&
                Quaternion.Angle(localRotation, expected.rotation) < 0.001f &&
                Vector3.Distance(localScale, expected.lossyScale) < 0.0001f,
                "Room overlay must use the saved scan transform in anchor coordinates");
            Require(Vector3.Distance(camera.transform.position, headPosition) < 0.0001f &&
                Quaternion.Angle(camera.transform.rotation, headRotation) < 0.001f,
                "Loading the scan must not move or recenter the wearer");
            camera.transform.SetPositionAndRotation(new Vector3(-8, 2, 11), Quaternion.Euler(35, 143, 0));
            Call(capture, "ApplyRoomPreviewTransform", scan, matrix);
            Require(Vector3.Distance(scan.localPosition, localPosition) < 0.0001f &&
                Quaternion.Angle(scan.localRotation, localRotation) < 0.001f,
                "Refreshing from another head position must not recenter the room");
            anchor.SetPositionAndRotation(new Vector3(-1, 0.8f, 7), Quaternion.Euler(0, -33, 0));
            Require(Vector3.Distance(scan.position, anchor.TransformPoint(localPosition)) < 0.0001f &&
                Vector3.Distance(scan.localPosition, localPosition) < 0.0001f &&
                Quaternion.Angle(scan.localRotation, localRotation) < 0.001f &&
                Vector3.Distance(scan.localScale, localScale) < 0.0001f,
                "Relocalizing the anchor must preserve the room's local alignment");
            Require(Controller.GetMethod("ToggleInspection", Private) == null &&
                Controller.GetMethod("UpdateInspectionMovement", Private) == null,
                "Room capture must not include a controller mode that relocates the scan");
            Debug.Log("PASS: Quest room overlay preserves anchor-relative pose through head movement, refresh and anchor relocalization; no wearer recenter or virtual scan movement");
        }

        private static void CheckExistingProjectFirstScan(QuestCaptureController capture, object bridge, Type profileType)
        {
            var staticPrivate = BindingFlags.Static | BindingFlags.NonPublic;
            var projectType = Controller.GetNestedType("ProjectInfo", BindingFlags.NonPublic);
            var sessionType = Controller.GetNestedType("SessionInfo", BindingFlags.NonPublic);
            var sessionsType = Controller.GetNestedType("ProjectSessions", BindingFlags.NonPublic);
            var projectId = Guid.NewGuid().ToString();
            var project = Activator.CreateInstance(projectType);
            projectType.GetField("id").SetValue(project, projectId);
            projectType.GetField("name").SetValue(project, "Room October fixture");
            var response = Activator.CreateInstance(sessionsType);
            var empty = Array.CreateInstance(sessionType, 0);
            sessionsType.GetField("sessions").SetValue(response, empty);
            var allowed = Controller.GetMethod("CanInitializeFirstScan", staticPrivate);
            var args = new object[] { project, response, null };
            Require((bool)allowed.Invoke(null, args), "An existing verified project with no photos must allow first setup");
            args[1] = null;
            Require(!(bool)allowed.Invoke(null, args), "An unavailable PC session list must block setup");
            args[1] = response;
            var first = Activator.CreateInstance(sessionType);
            var firstId = Guid.NewGuid().ToString();
            var firstOrigin = Guid.NewGuid().ToString();
            foreach (var pair in new[] { new[] { "id", firstId }, new[] { "project_id", projectId },
                new[] { "tracking_origin_id", firstOrigin }, new[] { "status", "capturing" } })
                sessionType.GetField(pair[0]).SetValue(first, pair[1]);
            var one = Array.CreateInstance(sessionType, 1); one.SetValue(first, 0);
            sessionsType.GetField("sessions").SetValue(response, one);
            Require((bool)allowed.Invoke(null, args), "An existing empty session must remain eligible");
            var select = Controller.GetMethod("SelectEmptyFirstScanSession", staticPrivate);
            Require(ReferenceEquals(select.Invoke(null, new object[] { response, firstOrigin }), first),
                "First setup must reuse an existing empty session and its tracking origin");
            sessionType.GetField("capture_count").SetValue(first, 1);
            Require(!(bool)allowed.Invoke(null, args), "Saved photos must block new first-scan origin creation");
            Require(select.Invoke(null, new object[] { response, firstOrigin }) == null,
                "A session containing photos must never be selected as empty");
            sessionType.GetField("capture_count").SetValue(first, -1);
            Require(!(bool)allowed.Invoke(null, args), "Invalid negative counts must block initialization");
            sessionType.GetField("capture_count").SetValue(first, 0);
            sessionType.GetField("project_id").SetValue(first, Guid.NewGuid().ToString());
            Require(!(bool)allowed.Invoke(null, args), "Cross-project PC responses must block initialization");
            sessionType.GetField("project_id").SetValue(first, projectId);
            sessionType.GetField("status").SetValue(first, "finished");
            Require(select.Invoke(null, new object[] { response, firstOrigin }) == null,
                "Finished sessions must stay untouched");
            var beforeConfig = Get(capture, "config");
            var beforeProfiles = JsonUtility.ToJson(Get(capture, "localProfiles"));
            var pathMethod = Controller.GetMethod("PendingProjectScanPath", staticPrivate);
            var pendingPath = (string)pathMethod.Invoke(null, new object[] { projectId });
            Require(!System.IO.File.Exists(pendingPath), "Fixture must use an unused retry-record path");
            try
            {
                Call(capture, "PrepareExistingProjectScan", project);
                var pending = Get(capture, "pendingProjectScan");
                var firstBytes = System.IO.File.ReadAllText(pendingPath);
                Require(JsonUtility.ToJson(pending).Contains(projectId) && !firstBytes.Contains("project_request_id"),
                    "Existing-project setup must persist only session retry IDs for the same project");
                Call(capture, "PrepareExistingProjectScan", project);
                Require(System.IO.File.ReadAllText(pendingPath) == firstBytes,
                    "Retry/reopen must reuse durable request IDs without duplicates");
                Require(ReferenceEquals(beforeConfig, Get(capture, "config")) &&
                    JsonUtility.ToJson(Get(capture, "localProfiles")) == beforeProfiles,
                    "Preparing/cancelling confirmation must not change active identity or local profiles");
                var selected = Get(capture, "initializeProject");
                Require((string)projectType.GetField("id").GetValue(selected) == projectId &&
                    (string)projectType.GetField("name").GetValue(selected) == "Room October fixture",
                    "Selected existing ID/name must survive initialization preparation");
            }
            finally { if (System.IO.File.Exists(pendingPath)) System.IO.File.Delete(pendingPath); }
            Debug.Log("PASS: Existing-project first scan: identity/name preserved, empty session reuse, durable retry/reopen, no preparation side effects, unavailable/captured/cross-project/invalid-count guards");
        }

        private static object Get(object target, string name) => Controller.GetField(name, Private).GetValue(target);
        private static void Set(object target, string name, object value) => Controller.GetField(name, Private).SetValue(target, value);
        private static void Call(object target, string name, params object[] args) => Controller.GetMethod(name, Private).Invoke(target, args);
        private static void Require(bool condition, string message)
        { if (!condition) throw new InvalidOperationException(message); }
    }
}
