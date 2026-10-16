using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Meta.XR;
using UnityEngine;
using UnityEngine.Networking;

namespace QuestSplatBridge
{
    /// <summary>
    /// Attach beside Meta's PassthroughCameraAccess in a Quest camera-rig scene.
    /// A stores a physical RGB camera frame plus its own timestamp, calibration and pose.
    /// B/X/Y display the saved Gaussian, mesh and organized model library.
    /// </summary>
    public sealed partial class QuestCaptureController : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraAccess cameraAccess;
        [SerializeField] private Transform sessionOrigin;
        [SerializeField] private Camera statusCamera;
        [SerializeField] private TextMesh statusText;
        [SerializeField] private Transform legendPanel;
        [SerializeField] private TextMesh controllerLegendText;
        [SerializeField, Range(1, 100)] private int maxQueuedCaptures = 40;
        [SerializeField] private float retryIntervalSeconds = 5f;

        // Set by the existing build owner from the contributor's app registration.
        // No maintainer account identity or platform credential is shipped in source.
        public static string MetaQuestAppId => Oculus.Platform.PlatformSettings.MobileAppID;
        private bool metaPlatformReady;
        private bool metaPlatformInitializing;
        private string metaPlatformStatus = "Verifying scanner app with Meta";

        private const string ConfigFile = "bridge.json";
        private const string SessionMarkerFile = "session-origin.json";
        private const string QueueFolder = "captures";
        private const long MaxImageBytes = 20L * 1024L * 1024L;

        private BridgeConfig config;
        private string queuePath;
        private bool capturePending;
        private bool uploadRunning;
        private bool originInvalidated;
        private bool sessionHasCapture;
        private bool recenterSubscribed;
        private bool projectVerified;
        private string projectName;
        private string status = "Camera starting";
        private string displayMode = "passthrough";
        private string requestedPreview;
        private string lastCapturedFrame = "";
        private GameObject shutterFlash;
        private Material shutterMaterial;
        private AudioSource shutterAudio;
        private AudioClip shutterClick;
        private Coroutine shutterAnimation;
        private enum HudDisplayMode { Full, Compact, Hidden }
        private HudDisplayMode hudMode = HudDisplayMode.Full;
        private bool menuPlaced;
        private bool menuUserPlaced;
        private Vector3 menuWorldPosition;
        private Quaternion menuWorldRotation;
        private OVRCameraRig menuCameraRig;
        private Transform menuBackdrop;
        private bool menuDragging;
        private bool menuGripHeld;
        private Vector3 menuGrabControllerStart;
        private Vector3 menuGrabPanelStart;
        private bool exitRequested;
        private bool boundaryHideRequested = true;
        private OVRManager boundaryManager;
        private OVRPassthroughLayer passthroughLayer;
#if DEVELOPMENT_BUILD || UNITY_EDITOR
        private float nextDebugCaptureCheck;
#endif

        [DllImport("OVRPlugin", CallingConvention = CallingConvention.Cdecl)]
        private static extern OVRPlugin.Result ovrp_GetNodePoseStateAtTime(
            double time, OVRPlugin.Node nodeId, out OVRPlugin.PoseStatef nodePoseState);

        private void InitializeMetaPlatform()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (metaPlatformReady || metaPlatformInitializing) return;
            if (!ulong.TryParse(MetaQuestAppId, out var appId) || appId == 0)
            {
                metaPlatformStatus = "Meta app registration is missing; rebuild with your app ID";
                status = metaPlatformStatus;
                return;
            }
            metaPlatformInitializing = true;
            metaPlatformStatus = "Verifying scanner app with Meta";
            Debug.Log("QuestMetaIdentity: scanner init requested; package=" + Application.identifier);
            try
            {
                Oculus.Platform.Core.AsyncInitialize(MetaQuestAppId).OnComplete(message => {
                    if (!this) return;
                    if (message.IsError || message.Data == null ||
                        message.Data.Result != Oculus.Platform.PlatformInitializeResult.Success)
                    {
                        metaPlatformInitializing = false;
                        metaPlatformStatus = "Meta scanner verification failed; A retries";
                        status = metaPlatformStatus;
                        Debug.LogWarning("QuestMetaIdentity: scanner initialization failed; code=" +
                            (message.IsError ? message.GetError().Code.ToString() : message.Data?.Result.ToString()));
                        return;
                    }
                    Debug.Log("QuestMetaIdentity: scanner SDK initialized; checking entitlement");
                    Oculus.Platform.Entitlements.IsUserEntitledToApplication().OnComplete(entitlement => {
                        if (!this) return;
                        metaPlatformInitializing = false;
                        metaPlatformReady = !entitlement.IsError;
                        if (!metaPlatformReady)
                        {
                            metaPlatformStatus = "Meta app access not verified; capture paused\nApp/package mapping must be checked";
                            status = metaPlatformStatus;
                            Debug.LogWarning("QuestMetaIdentity: scanner entitlement failed; code=" + entitlement.GetError().Code);
                            return;
                        }
                        metaPlatformStatus = "Meta scanner app verified";
                        Debug.Log("QuestMetaIdentity: scanner SDK and entitlement verified");
                        if (Configured() && projectVerified && !anchorSaved) InitializeSessionAnchor();
                    });
                });
            }
            catch (Exception error)
            {
                metaPlatformInitializing = false;
                metaPlatformStatus = "Meta scanner initialization unavailable; A retries";
                status = metaPlatformStatus;
                Debug.LogWarning("QuestMetaIdentity: scanner initialization exception " + error.GetType().Name);
            }
#else
            // Local editor fixtures exercise project/UI policy only, not Meta identity.
            metaPlatformReady = true;
#endif
        }

        private void Start()
        {
            queuePath = Path.Combine(Application.persistentDataPath, QueueFolder);
            Directory.CreateDirectory(queuePath);
            LoadConfig();
            InitializeMetaPlatform();
            InitializeProjects();
            LoadSessionState();
            if (cameraAccess == null || sessionOrigin == null)
            {
                status = "Set camera access and session origin";
                return;
            }

            boundaryManager = OVRManager.instance;
            passthroughLayer = FindObjectOfType<OVRPassthroughLayer>(true);
            if (legendPanel != null) menuBackdrop = legendPanel.Find("Controls contrast backdrop");
            menuCameraRig = statusCamera == null ? null : statusCamera.GetComponentInParent<OVRCameraRig>();
            InitializeShutterFeedback();

            // One permission owner in this app. Disable the sample's RequestPermissionsOnce
            // if this controller is imported into a sample scene.
            OVRPermissionsRequester.Request(new[]
            {
                OVRPermissionsRequester.Permission.PassthroughCameraAccess,
                OVRPermissionsRequester.Permission.Scene
            });
            StartCoroutine(WarmSpatialRaycaster());
            if (OVRManager.display != null)
            {
                OVRManager.display.RecenteredPose += OnRecentered;
                recenterSubscribed = true;
            }
            StartCoroutine(UploadLoop());
            verificationCoroutine = StartCoroutine(VerifyProjectLoop());
            StartCoroutine(PreviewRefreshLoop());
        }

        private void OnDestroy()
        {
            if (boundaryManager != null)
                boundaryManager.shouldBoundaryVisibilityBeSuppressed = false;
            DisposeLibraryModels();
            DisposePreview();
            DisposeSpatialRaycaster();
            if (shutterFlash != null) Destroy(shutterFlash);
            if (shutterMaterial != null) Destroy(shutterMaterial);
            if (shutterClick != null) Destroy(shutterClick);
            if (recenterSubscribed && OVRManager.display != null)
                OVRManager.display.RecenteredPose -= OnRecentered;
        }

        private void OnRecentered()
        {
            InvalidateProjectAreaResults();
            if (sessionAnchor != null)
            {
                anchorStableSince = -1f;
                anchorReadyAnnounced = false;
                capturePending = false;
                status = "Relocating scan; look around briefly";
                return;
            }
            // Quest may recenter during startup. Before the first captured view
            // there is no established physical frame to invalidate.
            if (!sessionHasCapture && !originInvalidated)
            {
                if (Configured()) status = "Ready after startup recenter";
                return;
            }
            originInvalidated = true;
            capturePending = false;
            displayMode = "passthrough";
            requestedPreview = null;
            status = "Tracking origin changed; create a new session";
        }

        private void Update()
        {
            if (cameraAccess == null || sessionOrigin == null) return;
            UpdateAnchorTracking();
#if DEVELOPMENT_BUILD || UNITY_EDITOR
            // ADB-only, one-shot test trigger. Use the same physical-camera path as
            // a button press; never supply fabricated pixels or tracking data.
            if (Time.unscaledTime >= nextDebugCaptureCheck)
            {
                nextDebugCaptureCheck = Time.unscaledTime + 1f;
                var requestPath = Path.Combine(Application.persistentDataPath, "capture-request.txt");
                if (File.Exists(requestPath))
                {
                    File.Delete(requestPath);
                    RequestCapture();
                    Debug.Log("Quest Splat: development capture requested");
                }
                var previewPath = Path.Combine(Application.persistentDataPath, "preview-request.txt");
                if (File.Exists(previewPath))
                {
                    var requestedRepresentation = File.ReadAllText(previewPath).Trim() == "depth" ? "depth" : "splat";
                    File.Delete(previewPath);
                    HideLibraryModel();
                    Preview(requestedRepresentation);
                }
                var controlPath = Path.Combine(Application.persistentDataPath, "control-request.txt");
                if (File.Exists(controlPath))
                {
                    var control = File.ReadAllText(controlPath).Trim();
                    File.Delete(controlPath);
                    if (int.TryParse(control, out var controlIndex) && controlIndex >= 0 && controlIndex < 5)
                        InvokeControl(controlIndex);
                }
            }
#endif

            if (OVRManager.hasInputFocus)
            {
                if (OVRInput.GetDown(OVRInput.RawButton.A, OVRInput.Controller.RTouch)) InvokeControl(0);
                if (OVRInput.GetDown(OVRInput.RawButton.B, OVRInput.Controller.RTouch)) InvokeControl(1);
                if (OVRInput.GetDown(OVRInput.RawButton.X, OVRInput.Controller.LTouch)) InvokeControl(2);
                if (OVRInput.GetDown(OVRInput.RawButton.Y, OVRInput.Controller.LTouch)) InvokeControl(3);
                if (OVRInput.GetDown(OVRInput.RawButton.Start, OVRInput.Controller.LTouch) ||
                    OVRInput.GetDown(OVRInput.Button.Start, OVRInput.Controller.LTouch)) ExitToHome();
                if (OVRInput.GetDown(OVRInput.RawButton.RThumbstick, OVRInput.Controller.RTouch))
                {
                    boundaryHideRequested = !boundaryHideRequested;
                    ApplyBoundaryVisibility();
                    status = boundaryHideRequested
                        ? "Boundary hide requested"
                        : "Boundary shown";
                }
                if (OVRInput.GetDown(OVRInput.RawButton.LThumbstick, OVRInput.Controller.LTouch))
                    CycleHudMode();
                if (!IsLibraryOpen && OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
                {
                    HideLibraryModel();
                    Preview("depth");
                }
                UpdateMenuGrab();
            }
            else
            {
                // A lost focus/controller sample must not turn into a large drag
                // when tracking resumes. Require a fresh grip press.
                menuDragging = false;
                menuGripHeld = true;
                if (projectsInArea.Count > 0) InvalidateProjectAreaResults();
            }
            // The SDK reports these four values for the same current camera image.
            // Wait for a fresh frame after the button press; do not photograph a UI screenshot.
            if (capturePending && cameraAccess.IsPlaying && cameraAccess.IsUpdatedThisFrame)
            {
                capturePending = false;
                CaptureCurrentFrame();
            }
            else if (capturePending && !cameraAccess.IsPlaying)
            {
                status = "Waiting for headset camera";
            }
        }

        private void LateUpdate()
        {
            if (statusText == null || statusCamera == null) return;
            ApplyBoundaryVisibility();
            var compact = hudMode == HudDisplayMode.Compact;
            UpdateHudPlacement();
            if (menuBackdrop != null)
            {
                menuBackdrop.localPosition = compact ? new Vector3(0, -0.035f, 0.025f)
                    : new Vector3(0, -0.20f, 0.025f);
                menuBackdrop.localScale = compact ? new Vector3(0.66f, 0.11f, 1f)
                    : new Vector3(0.64f, 0.50f, 1f);
            }
            if (controllerLegendText != null)
            {
                controllerLegendText.gameObject.SetActive(!compact);
                controllerLegendText.text = IsLibraryOpen
                        ? "A Previous   B Next\nX Open   Y Back\nL click Cycle  Menu Exit"
                        : requestedPreview == "splat" && previewAligned
                            ? "A Capture   B Splat\nX Mesh      Y Projects\nR trigger Depth  Menu Exit\nGrip HUD  L click Cycle"
                            : "A Capture   B Splat\nX Mesh      Y Projects\nR trigger Depth  Menu Exit\nGrip HUD  L click Cycle";
            }
            if (compact)
            {
                statusText.transform.localPosition = new Vector3(0, -0.02f, 0);
                var mode = IsLibraryOpen ? "Models" :
                    requestedPreview == "mesh" ? "Mesh" : requestedPreview == "depth" ? "Depth" : requestedPreview == "splat" ? "Splat" : "Camera";
                statusText.text = previewTotalViews > 0
                    ? (IsRenderedPreview(requestedPreview) && IsPreviewSelected
                        ? mode + "  " + previewAcceptedViews + " / " + previewTotalViews + " photos"
                        : mode + "  " + previewTotalViews + " saved photos")
                    : mode + "  waiting for scan counts";
                return;
            }
            statusText.transform.localPosition = new Vector3(0,
                IsLibraryOpen ? -0.16f : -0.205f, 0);
            var boundaryState = boundaryManager == null ? "Boundary state unavailable" :
                boundaryManager.isBoundaryVisibilitySuppressed ? "Boundary hidden" :
                boundaryManager.shouldBoundaryVisibilityBeSuppressed ? "Boundary still visible (Quest)" :
                boundaryHideRequested ? "Boundary waiting for passthrough" : "Boundary shown";
            statusText.text = WrapHud(projectVerified ? projectName : "Checking PC project", 28) + "\n" +
                WrapHud(CurrentMenuStatus, 28) + (IsLibraryOpen ? "" : "\n" + WrapHud(boundaryState, 28));
        }

        private static string WrapHud(string value, int width)
        {
            var result = new StringBuilder();
            var column = 0;
            foreach (var line in (value ?? "").Replace("\r", "").Split('\n'))
            {
                if (result.Length > 0) result.Append('\n');
                column = 0;
                foreach (var word in line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    if (column > 0 && column + word.Length + 1 > width)
                    { result.Append('\n'); column = 0; }
                    else if (column > 0) { result.Append(' '); column++; }
                    for (var start = 0; start < word.Length; start += width)
                    {
                        if (start > 0) { result.Append('\n'); column = 0; }
                        var part = word.Substring(start, Math.Min(width, word.Length - start));
                        result.Append(part); column += part.Length;
                    }
                }
            }
            return result.ToString();
        }

        private void CycleHudMode()
        {
            SetHudMode((HudDisplayMode)(((int)hudMode + 1) % 3));
            if (hudMode == HudDisplayMode.Full) EnsureFullMenuVisible();
        }

        // Explicit menu access must recover a panel left behind the wearer.
        // Keep a visible user placement; only bring an inaccessible panel back.
        private void ShowFullMenu()
        {
            SetHudMode(HudDisplayMode.Full);
            EnsureFullMenuVisible();
        }

        private void EnsureFullMenuVisible()
        {
            if (legendPanel == null || statusCamera == null) return;
            EnsureFullMenuPlacement();
            var center = statusCamera.WorldToViewportPoint(
                legendPanel.TransformPoint(new Vector3(0, -0.20f, 0)));
            if (center.z < 0.4f || center.z > 2.5f || center.x < 0.15f || center.x > 0.85f ||
                center.y < 0.15f || center.y > 0.85f ||
                Vector3.Dot(legendPanel.forward, statusCamera.transform.forward) < 0.3f)
            {
                menuUserPlaced = false;
                CenterFullMenu();
            }
        }

        private void ReportCaptureIssue(string message)
        {
            status = message;
            ShowFullMenu();
            Debug.LogWarning("Quest Capture blocked: " + message);
        }

        private void SetHudMode(HudDisplayMode mode)
        {
            hudMode = mode;
            menuDragging = false;
            if (legendPanel != null) legendPanel.gameObject.SetActive(hudMode != HudDisplayMode.Hidden);
            // Compact uses the same renderer, but never overwrites the remembered
            // full-menu pose. Reopening any menu restores that exact world pose.
            if (mode == HudDisplayMode.Full && menuPlaced)
                RestoreFullMenuPlacement();
        }

        private void EnsureFullMenuPlacement()
        {
            if (menuPlaced || legendPanel == null || statusCamera == null) return;
            CenterFullMenu();
        }

        private void CenterFullMenu()
        {
            var head = statusCamera.transform;
            var upright = Mathf.Abs(Vector3.Dot(head.forward, Vector3.up)) < 0.98f
                ? Vector3.up : head.up;
            menuWorldRotation = Quaternion.LookRotation(head.forward, upright);
            // The text group has a top pivot; center the panel itself on the
            // gaze ray, not its top edge. Avoid carrying head roll into the panel.
            menuWorldPosition = head.position + head.forward * 0.90f +
                menuWorldRotation * new Vector3(0, 0.20f, 0);
            menuPlaced = true;
            RestoreFullMenuPlacement();
        }

        private void RestoreFullMenuPlacement()
        {
            if (legendPanel != null && menuPlaced)
                legendPanel.SetPositionAndRotation(menuWorldPosition, menuWorldRotation);
        }

        private void UpdateHudPlacement()
        {
            if (legendPanel == null || statusCamera == null) return;
            if (hudMode == HudDisplayMode.Compact)
            {
                var head = statusCamera.transform;
                legendPanel.SetPositionAndRotation(
                    head.position + head.forward * 0.90f + head.up * 0.31f, head.rotation);
            }
            else if (hudMode == HudDisplayMode.Full)
            {
                // Do not fix the initial pose while Quest is still at a startup
                // origin or its system menu has focus.
                if (!menuUserPlaced && OVRManager.hasInputFocus && OVRManager.isHmdPresent)
                    CenterFullMenu();
                RestoreFullMenuPlacement();
            }
        }

        private void UpdateMenuGrab()
        {
            if (legendPanel == null || statusCamera == null) return;
            var grip = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,
                OVRInput.Controller.RTouch) > 0.55f;
            if (hudMode != HudDisplayMode.Full)
            {
                if (!grip || menuGripHeld) { menuGripHeld = grip; return; }
                ShowFullMenu();
            }
            if (menuCameraRig == null || menuCameraRig.trackingSpace == null ||
                !OVRInput.GetControllerPositionTracked(OVRInput.Controller.RTouch))
            {
                menuDragging = false;
                menuGripHeld = grip;
                return;
            }
            // OVRInput is relative to the camera rig's tracking space, not the
            // saved scan anchor (sessionOrigin changes after relocalization).
            var controllerPosition = menuCameraRig.trackingSpace.TransformPoint(
                OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch));
            if (!menuDragging && grip && !menuGripHeld)
                BeginMenuGrab(controllerPosition);
            if (menuDragging && grip)
                MoveMenuGrab(controllerPosition);
            else if (menuDragging)
                menuDragging = false;
            menuGripHeld = grip;
        }

        private void BeginMenuGrab(Vector3 controllerPosition)
        {
            EnsureFullMenuPlacement();
            menuUserPlaced = true;
            menuDragging = true;
            // Pickup has no teleport or wrist-driven rotation. Move only by
            // the hand's world translation from the current saved placement.
            menuGrabPanelStart = menuWorldPosition;
            menuGrabControllerStart = controllerPosition;
            status = "HUD grabbed; release grip to place it";
        }

        private void MoveMenuGrab(Vector3 controllerPosition)
        {
            menuWorldPosition = menuGrabPanelStart + (controllerPosition - menuGrabControllerStart);
            RestoreFullMenuPlacement();
        }

        private void ApplyBoundaryVisibility()
        {
            if (boundaryManager == null) boundaryManager = OVRManager.instance;
            if (boundaryManager == null) return;
            if (passthroughLayer == null)
                passthroughLayer = FindObjectOfType<OVRPassthroughLayer>(true);
            // The OS honors this request only with active passthrough. Allow the
            // room-registered splat over passthrough, but restore the boundary for
            // unaligned previews, library models, and menus.
            boundaryManager.shouldBoundaryVisibilityBeSuppressed = boundaryHideRequested &&
                passthroughLayer != null && passthroughLayer.isActiveAndEnabled &&
                OVRManager.hasInputFocus && !exitRequested &&
                (requestedPreview == null || (IsRenderedPreview(requestedPreview) && previewAligned)) &&
                !IsLibraryOpen &&
                (libraryModelObject == null || !libraryModelObject.activeInHierarchy);
        }

        private void RequestCapture()
        {
            if (exitRequested || capturePending || projectSwitching || IsLibraryOpen) return;
            if (!metaPlatformReady)
            {
                ReportCaptureIssue(metaPlatformStatus);
                InitializeMetaPlatform();
                return;
            }
            if (Configured() && projectVerified && !anchorSaved)
            {
                if (anchorInitializationGeneration == configGeneration) return;
                ShowFullMenu();
                OVRPermissionsRequester.Request(new[] { OVRPermissionsRequester.Permission.Scene });
                InitializeSessionAnchor();
                return;
            }
            capturePending = true;
        }

        private void ExitToHome()
        {
            if (exitRequested) return;
            exitRequested = true;
            capturePending = false;
            // Every saved capture is already on disk. Interrupted network requests
            // remain queued and are retried with the same identity on the next launch.
            status = "Exiting; unsent photos remain saved";
            Application.Quit();
        }

        private void CaptureCurrentFrame()
        {
            try
            {
                if (projectSwitching || IsLibraryOpen) return;
                if (!metaPlatformReady) { status = metaPlatformStatus; return; }
                // A protocol capture must already belong to a real PC project and
                // session. Do not save image/metadata pairs with empty identities.
                if (!Configured())
                {
                    ReportCaptureIssue("Pair PC project/session before capture");
                    return;
                }
                if (!projectVerified)
                {
                    ReportCaptureIssue("Confirm project with PC before capture");
                    return;
                }
                if (originInvalidated)
                {
                    ReportCaptureIssue("New capture session needed after recenter");
                    return;
                }
                if (!AnchorReady)
                {
                    ReportCaptureIssue("Relocating scan; look around briefly");
                    return;
                }
                if (!OVRPermissionsRequester.IsPermissionGranted(
                        OVRPermissionsRequester.Permission.PassthroughCameraAccess))
                {
                    ReportCaptureIssue("Camera permission required");
                    return;
                }
                if (!PassthroughCameraAccess.IsSupported || !cameraAccess.IsPlaying)
                {
                    ReportCaptureIssue("Camera unavailable");
                    return;
                }
                if (!ovrp_GetNodePoseStateAtTime(OVRPlugin.GetTimeInSeconds(),
                        OVRPlugin.Node.Head, out _).IsSuccess())
                {
                    ReportCaptureIssue("Tracking unavailable; capture skipped");
                    return;
                }
                if (CaptureMetadataPaths(queuePath).Length >= maxQueuedCaptures)
                {
                    ReportCaptureIssue("Capture queue full; connect to PC");
                    return;
                }

                var resolution = cameraAccess.CurrentResolution;
                var intrinsics = cameraAccess.Intrinsics;
                var timestamp = cameraAccess.Timestamp;
                var pose = cameraAccess.GetCameraPose();
                if (resolution.x <= 0 || resolution.y <= 0 || timestamp == default ||
                    intrinsics.SensorResolution.x <= 0 || intrinsics.SensorResolution.y <= 0 ||
                    !Finite(intrinsics.FocalLength.x) || !Finite(intrinsics.FocalLength.y) ||
                    intrinsics.FocalLength.x <= 0 || intrinsics.FocalLength.y <= 0 ||
                    !Finite(intrinsics.PrincipalPoint.x) || !Finite(intrinsics.PrincipalPoint.y) ||
                    !Finite(pose.position.x) || !Finite(pose.position.y) || !Finite(pose.position.z) ||
                    !Finite(pose.rotation.x) || !Finite(pose.rotation.y) ||
                    !Finite(pose.rotation.z) || !Finite(pose.rotation.w))
                {
                    ReportCaptureIssue("Invalid frame calibration or pose");
                    return;
                }
                var rotationNorm = Mathf.Sqrt(pose.rotation.x * pose.rotation.x +
                    pose.rotation.y * pose.rotation.y + pose.rotation.z * pose.rotation.z +
                    pose.rotation.w * pose.rotation.w);
                if (!Finite(rotationNorm) || rotationNorm < 0.5f || rotationNorm > 1.5f)
                {
                    ReportCaptureIssue("Invalid tracking rotation; capture skipped");
                    return;
                }
                // Normalize the SDK rotation before constructing the rigid pose.
                var sessionFromCamera = sessionOrigin.worldToLocalMatrix *
                    Matrix4x4.TRS(pose.position, pose.rotation.normalized, Vector3.one);
                var frameKey = timestamp.Ticks.ToString(CultureInfo.InvariantCulture);
                if (lastCapturedFrame == frameKey)
                {
                    ReportCaptureIssue("Wait for a new camera frame");
                    return;
                }

                // GetColors reads Meta's RGB camera pixels, never the rendered eye/UI view.
                // Keep timestamp/pose captured above before the expensive JPEG conversion.
                var triggerRealtime = Time.realtimeSinceStartupAsDouble;
                var colors = cameraAccess.GetColors();
                if (!colors.IsCreated || colors.Length < resolution.x * resolution.y)
                {
                    ReportCaptureIssue("Camera pixels unavailable");
                    return;
                }
                if (cameraAccess.Timestamp.Ticks != timestamp.Ticks)
                {
                    ReportCaptureIssue("Camera frame changed during read; retry");
                    return;
                }
                var captureId = Guid.NewGuid().ToString();
                var spatialBytes = CaptureSpatialEvidence(captureId, timestamp, triggerRealtime);
                var texture = new Texture2D(resolution.x, resolution.y, TextureFormat.RGBA32, false);
                // Match Meta's CameraToWorld sample. GetColors returns a native
                // raw readback buffer, which may be larger than the pixel count.
                texture.LoadRawTextureData(colors);
                texture.Apply(false, false);
                var jpeg = texture.EncodeToJPG(95);
                Destroy(texture);
                if (jpeg == null || jpeg.Length == 0 || jpeg.LongLength > MaxImageBytes)
                {
                    ReportCaptureIssue("Image encoding failed");
                    return;
                }

                // Match MRUK85 CalcSensorCropRegion: center-crop to output aspect,
                // then uniformly scale. Independent X/Y scaling distorted focal lengths.
                var pixelScale = Mathf.Max((float)resolution.x / intrinsics.SensorResolution.x,
                    (float)resolution.y / intrinsics.SensorResolution.y);
                var cropX = (intrinsics.SensorResolution.x - resolution.x / pixelScale) * 0.5f;
                var cropY = (intrinsics.SensorResolution.y - resolution.y / pixelScale) * 0.5f;
                var metadata = new CaptureMetadata
                {
                    capture_id = captureId,
                    project_id = config.project_id,
                    session_id = config.session_id,
                    tracking_origin_id = config.tracking_origin_id,
                    camera_id = cameraAccess.CameraPosition == PassthroughCameraAccess.CameraPositionType.Left ? "left" : "right",
                    image_timestamp_raw = timestamp.ToString("O", CultureInfo.InvariantCulture),
                    image_timestamp_kind = timestamp.Kind.ToString(),
                    timestamp_clock = "meta_datetime",
                    image_width = resolution.x,
                    image_height = resolution.y,
                    image_encoding = "jpeg",
                    image_sha256 = Sha256(jpeg),
                    spatial_evidence_sha256 = spatialBytes == null ? null : Sha256(spatialBytes),
                    intrinsics = new Intrinsics
                    {
                        fx = intrinsics.FocalLength.x * pixelScale,
                        fy = intrinsics.FocalLength.y * pixelScale,
                        cx = (intrinsics.PrincipalPoint.x - cropX) * pixelScale,
                        cy = (intrinsics.PrincipalPoint.y - cropY) * pixelScale
                    },
                    camera_to_session = RowMajor(sessionFromCamera),
                    coordinate_convention = "unity_lh_xright_yup_zforward",
                    position_units = "meters",
                    tracking_valid = true,
                    image_orientation = "unity_texture_native",
                    calibration_revision = "mruk85_center_crop_v1",
                    capture_source = "quest_camera"
                };
                var path = Path.Combine(queuePath, captureId);
                // Image first; metadata is the queue commit marker. A crash before the
                // metadata write leaves only an orphan image, never a false queued capture.
                File.WriteAllBytes(path + ".jpg", jpeg);
                if (spatialBytes != null) File.WriteAllBytes(SpatialPath(path + ".json"), spatialBytes);
                File.WriteAllText(path + ".json.tmp", SerializeCaptureMetadata(metadata));
                File.Move(path + ".json.tmp", path + ".json");
                MarkSessionCaptured();
                lastCapturedFrame = frameKey;
                PlayShutterFeedback();
                OVRInput.SetControllerVibration(0.2f, 0.15f, OVRInput.Controller.RTouch);
                StartCoroutine(StopHaptic());
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                ReportCaptureIssue("Capture failed: " + ex.GetType().Name);
            }
        }

        private IEnumerator StopHaptic()
        {
            yield return new WaitForSeconds(0.08f);
            OVRInput.SetControllerVibration(0, 0, OVRInput.Controller.RTouch);
        }

        private void InitializeShutterFeedback()
        {
            if (statusCamera == null) return;
            shutterFlash = GameObject.CreatePrimitive(PrimitiveType.Quad);
            shutterFlash.name = "Capture shutter flash";
            shutterFlash.transform.SetParent(statusCamera.transform, false);
            const float distance = 0.65f;
            shutterFlash.transform.localPosition = new Vector3(0, 0, distance);
            shutterFlash.transform.localRotation = Quaternion.Euler(0, 180, 0);
            var height = 2f * distance * Mathf.Tan(statusCamera.fieldOfView * Mathf.Deg2Rad * 0.5f);
            shutterFlash.transform.localScale = new Vector3(height * statusCamera.aspect, height, 1);
            Destroy(shutterFlash.GetComponent<Collider>());
            var shader = Shader.Find("Unlit/Transparent");
            if (shader == null)
            {
                Destroy(shutterFlash);
                shutterFlash = null;
            }
            else
            {
                shutterMaterial = new Material(shader);
                shutterMaterial.color = new Color(0.85f, 0.9f, 1f, 0.44f);
                shutterMaterial.renderQueue = 3999;
                shutterFlash.GetComponent<MeshRenderer>().sharedMaterial = shutterMaterial;
            }
            if (shutterFlash != null) shutterFlash.SetActive(false);

            shutterAudio = statusCamera.gameObject.AddComponent<AudioSource>();
            shutterAudio.playOnAwake = false;
            shutterAudio.spatialBlend = 0f;
            shutterAudio.volume = 0.38f;
            const int rate = 24000;
            var samples = new float[rate / 16];
            var noise = new System.Random(17);
            for (var i = 0; i < samples.Length; i++)
            {
                var envelope = Mathf.Exp(-i / 250f);
                samples[i] = (float)(noise.NextDouble() * 2.0 - 1.0) * envelope * 0.6f;
            }
            shutterClick = AudioClip.Create("Capture click", samples.Length, 1, rate, false);
            shutterClick.SetData(samples, 0);
        }

        private void PlayShutterFeedback()
        {
            if (shutterAudio != null && shutterClick != null)
                shutterAudio.PlayOneShot(shutterClick);
            if (shutterFlash == null) return;
            if (shutterAnimation != null) StopCoroutine(shutterAnimation);
            shutterAnimation = StartCoroutine(FlashShutter());
        }

        private IEnumerator FlashShutter()
        {
            shutterFlash.SetActive(true);
            yield return new WaitForSecondsRealtime(0.065f);
            if (shutterFlash != null) shutterFlash.SetActive(false);
            shutterAnimation = null;
        }

        private IEnumerator VerifyProjectLoop()
        {
            while (!projectVerified)
            {
                if (Configured()) yield return VerifyProjectOnce();
                if (!projectVerified) yield return new WaitForSeconds(retryIntervalSeconds);
            }
        }

        private IEnumerator VerifyProjectOnce()
        {
            var generation = configGeneration;
            foreach (var endpoint in Endpoints())
            {
                var root = endpoint.TrimEnd('/') + "/api/v1/";
                using (var sessionRequest = UnityWebRequest.Get(root + "sessions/" +
                           UnityWebRequest.EscapeURL(config.session_id)))
                {
                    ConfigureCertificate(sessionRequest);
                    sessionRequest.timeout = 5;
                    sessionRequest.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return sessionRequest.SendWebRequest();
                    if (generation != configGeneration) yield break;
                    if (sessionRequest.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (sessionRequest.result != UnityWebRequest.Result.Success)
                    {
                        status = "PC session verification failed";
                        yield break;
                    }
                    SessionInfo session;
                    try { session = JsonUtility.FromJson<SessionInfo>(sessionRequest.downloadHandler.text); }
                    catch (Exception) { session = null; }
                    if (session == null || session.id != config.session_id ||
                        session.project_id != config.project_id ||
                        session.tracking_origin_id != config.tracking_origin_id ||
                        session.status != "capturing")
                    {
                        status = "PC project/session mismatch; capture blocked";
                        yield break;
                    }
                    // The PC owns the authoritative count. A missing local anchor
                    // must never silently mint a new frame for a scanned session.
                    sessionHasCapture |= session.capture_count > 0;
                    previewTotalViews = Mathf.Max(previewTotalViews, session.capture_count);
                }
                using (var projectRequest = UnityWebRequest.Get(root + "projects/" +
                           UnityWebRequest.EscapeURL(config.project_id)))
                {
                    ConfigureCertificate(projectRequest);
                    projectRequest.timeout = 5;
                    projectRequest.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return projectRequest.SendWebRequest();
                    if (generation != configGeneration) yield break;
                    if (projectRequest.result != UnityWebRequest.Result.Success)
                    {
                        status = "PC project verification failed";
                        yield break;
                    }
                    ProjectInfo project;
                    try { project = JsonUtility.FromJson<ProjectInfo>(projectRequest.downloadHandler.text); }
                    catch (Exception) { project = null; }
                    if (project == null || project.id != config.project_id ||
                        string.IsNullOrWhiteSpace(project.name))
                    {
                        status = "PC project mismatch; capture blocked";
                        yield break;
                    }
                    projectName = project.name.Replace('\r', ' ').Replace('\n', ' ').Trim();
                    if (projectName.Length > 32) projectName = projectName.Substring(0, 29) + "...";
                    projectVerified = true;
                    RememberActiveCaptureState();
                    status = "Project confirmed; A captures";
                    if (sessionAnchor == null && !anchorSaved) InitializeSessionAnchor();
                    yield break;
                }
            }
        }

        private IEnumerator UploadLoop()
        {
            while (true)
            {
                var hadQueuedCaptures = false;
                if (!uploadRunning && Configured())
                {
                    foreach (var journal in Directory.GetFiles(queuePath, "*.received"))
                    {
                        try { CaptureTransferCleanup.CompleteAcknowledged(journal); }
                        catch (Exception error)
                        { Debug.LogException(error); status = "PC received image; local cleanup will retry"; }
                    }
                    var files = CaptureMetadataPaths(queuePath);
                    hadQueuedCaptures = files.Length > 0;
                    Array.Sort(files, StringComparer.Ordinal);
                    foreach (var file in files)
                    {
                        if (File.Exists(Path.ChangeExtension(file, ".received"))) continue;
                        yield return UploadOne(file);
                    }
                }
                // Idle polling makes a newly saved photo start sending on the next
                // few frames; failed uploads still use the longer retry interval.
                yield return new WaitForSeconds(hadQueuedCaptures ? retryIntervalSeconds : 0.1f);
            }
        }

        private IEnumerator UploadOne(string metadataPath)
        {
            uploadRunning = true;
            try
            {
            var imagePath = Path.ChangeExtension(metadataPath, ".jpg");
            if (!File.Exists(imagePath))
            {
                status = "Queue image missing";
                uploadRunning = false;
                yield break;
            }
            string metadataJson = null;
            CaptureMetadata metadata = null;
            byte[] image = null;
            byte[] spatial = null;
            try
            {
                metadataJson = File.ReadAllText(metadataPath);
                metadata = JsonUtility.FromJson<CaptureMetadata>(metadataJson);
                image = File.ReadAllBytes(imagePath);
                if (!string.IsNullOrEmpty(metadata?.spatial_evidence_sha256))
                    spatial = File.ReadAllBytes(SpatialPath(metadataPath));
            }
            catch (Exception error) { Debug.LogException(error); }
            if (metadata == null || image == null)
            { status = "Unreadable capture preserved; continuing queue"; yield break; }
            if (!string.IsNullOrEmpty(metadata.spatial_evidence_sha256) &&
                (spatial == null || spatial.Length > 2 * 1024 * 1024 ||
                 Sha256(spatial) != metadata.spatial_evidence_sha256))
            { status = "Spatial evidence missing or changed; capture preserved"; yield break; }
            if (metadata.session_id != config.session_id || metadata.project_id != config.project_id ||
                metadata.tracking_origin_id != config.tracking_origin_id)
            {
                status = "Queued capture belongs to another session";
                uploadRunning = false;
                yield break;
            }
            var form = new WWWForm();
            form.AddField("metadata_json", metadataJson);
            form.AddBinaryData("image", image, "capture.jpg", "image/jpeg");
            if (spatial != null)
                form.AddBinaryData("spatial_json", spatial, "capture.spatial.json", "application/json");
            foreach (var endpoint in Endpoints())
            {
                if (spatial != null)
                {
                    using (var health = UnityWebRequest.Get(endpoint.TrimEnd('/') + "/health"))
                    {
                        ConfigureCertificate(health);
                        health.timeout = 5;
                        health.redirectLimit = 0;
                        health.SetRequestHeader("Authorization", "Bearer " + config.token);
                        yield return health.SendWebRequest();
                        BridgeHealth capability = null;
                        if (health.result == UnityWebRequest.Result.Success)
                        {
                            try { capability = JsonUtility.FromJson<BridgeHealth>(health.downloadHandler.text); }
                            catch (Exception) { }
                        }
                        if (capability?.capabilities == null ||
                            Array.IndexOf(capability.capabilities, "spatial_evidence") < 0)
                            continue;
                    }
                }
                var url = endpoint.TrimEnd('/') + "/api/v1/sessions/" +
                          UnityWebRequest.EscapeURL(config.session_id) + "/captures";
                using (var request = UnityWebRequest.Post(url, form))
                {
                    ConfigureCertificate(request);
                    request.timeout = 8;
                    request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (request.result == UnityWebRequest.Result.Success)
                    {
                        try
                        {
                            var receipt = JsonUtility.FromJson<CaptureTransferReceipt>(request.downloadHandler.text);
                            var cleaned = CaptureTransferCleanup.Accept(metadataPath, metadata.capture_id,
                                metadata.project_id, metadata.session_id, metadata.image_sha256,
                                metadata.spatial_evidence_sha256, receipt);
                            if (!cleaned) status = "Receipt mismatch; image remains queued";
                        }
                        catch (Exception error)
                        {
                            Debug.LogException(error);
                            status = "Image retained; receipt or cleanup will retry";
                        }
                    }
                    else status = "Upload rejected; image queued (" + request.responseCode + ")";
                    uploadRunning = false;
                    yield break;
                }
            }
            status = spatial != null ? "PC spatial receiver unavailable; capture queued" :
                "PC unreachable on USB and Wi-Fi; image queued";
            uploadRunning = false;
            }
            finally { uploadRunning = false; }
        }

        private void LoadConfig()
        {
            var path = Path.Combine(Application.persistentDataPath, ConfigFile);
            if (!File.Exists(path))
            {
                status = "bridge.json needed before capture";
                config = new BridgeConfig();
                return;
            }
            try { config = JsonUtility.FromJson<BridgeConfig>(File.ReadAllText(path)); }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                status = "bridge.json invalid; capture unavailable";
                config = new BridgeConfig();
            }
        }

        private void LoadSessionState()
        {
            if (!Configured()) return;
            var markerPath = Path.Combine(Application.persistentDataPath, SessionMarkerFile);
            if (File.Exists(markerPath))
            {
                try
                {
                    var marker = JsonUtility.FromJson<SessionMarker>(File.ReadAllText(markerPath));
                    sessionHasCapture = marker != null &&
                        marker.project_id == config.project_id &&
                        marker.session_id == config.session_id &&
                        marker.tracking_origin_id == config.tracking_origin_id;
                }
                catch (Exception ex) { Debug.LogException(ex); }
            }
            if (!sessionHasCapture)
            {
                foreach (var path in CaptureMetadataPaths(queuePath))
                {
                    try
                    {
                        var capture = JsonUtility.FromJson<CaptureMetadata>(File.ReadAllText(path));
                        if (capture != null && capture.project_id == config.project_id &&
                            capture.session_id == config.session_id &&
                            capture.tracking_origin_id == config.tracking_origin_id)
                        {
                            sessionHasCapture = true;
                            break;
                        }
                    }
                    catch (Exception ex) { Debug.LogException(ex); }
                }
            }
            var perSessionMarker = Path.Combine(Application.persistentDataPath,
                "session-captured-" + config.session_id + ".json");
            if (File.Exists(perSessionMarker)) sessionHasCapture = true;
            if (localProfiles.TryGetValue(config.project_id, out var profile) &&
                profile.session_id == config.session_id &&
                profile.tracking_origin_id == config.tracking_origin_id && profile.has_capture)
                sessionHasCapture = true;
            if (sessionHasCapture)
            {
                originInvalidated = true;
                status = "New session needed after app restart";
            }
        }

        private void MarkSessionCaptured()
        {
            if (sessionHasCapture) return;
            var path = Path.Combine(Application.persistentDataPath, SessionMarkerFile);
            var marker = new SessionMarker
            {
                project_id = config.project_id,
                session_id = config.session_id,
                tracking_origin_id = config.tracking_origin_id
            };
            File.WriteAllText(path + ".tmp", JsonUtility.ToJson(marker));
            if (File.Exists(path)) File.Delete(path);
            File.Move(path + ".tmp", path);
            sessionHasCapture = true;
            try
            {
                AtomicWrite(Path.Combine(Application.persistentDataPath,
                    "session-captured-" + config.session_id + ".json"), JsonUtility.ToJson(marker));
                RememberActiveCaptureState();
            }
            catch (Exception error) { Debug.LogException(error); }
        }

        private bool Configured()
        {
            if (config == null || string.IsNullOrWhiteSpace(config.base_url) ||
                string.IsNullOrWhiteSpace(config.token) ||
                !Guid.TryParse(config.project_id, out _) ||
                !Guid.TryParse(config.session_id, out _) ||
                !Guid.TryParse(config.tracking_origin_id, out _)) return false;
            if (!Uri.TryCreate(config.base_url, UriKind.Absolute, out var uri) ||
                uri.Scheme != Uri.UriSchemeHttps) return false;
            if (!string.IsNullOrWhiteSpace(config.wifi_base_url) &&
                (!Uri.TryCreate(config.wifi_base_url, UriKind.Absolute, out var wifiUri) ||
                 wifiUri.Scheme != Uri.UriSchemeHttps)) return false;
            if (!string.IsNullOrEmpty(config.certificate_sha256) &&
                !ValidSha256(config.certificate_sha256)) return false;
            return true;
        }

        private string[] Endpoints()
        {
            if (string.IsNullOrWhiteSpace(config.wifi_base_url) ||
                string.Equals(config.base_url.TrimEnd('/'), config.wifi_base_url.TrimEnd('/'),
                    StringComparison.OrdinalIgnoreCase))
                return new[] { config.base_url };
            return new[] { config.base_url, config.wifi_base_url };
        }

        private void ConfigureCertificate(UnityWebRequest request)
        {
            // Authenticated bridge requests must stay at their selected endpoint.
            // Apply this here so session/project and preview requests cannot follow redirects.
            request.redirectLimit = 0;
            // With no pin, Unity's normal platform certificate validation applies.
            // For a private self-signed LAN certificate, pair out of band and pin its
            // exact DER SHA-256 fingerprint. Never accept every certificate.
            if (!string.IsNullOrEmpty(config.certificate_sha256))
                request.certificateHandler = new PinnedCertificateHandler(config.certificate_sha256);
        }

        private static bool ValidSha256(string value)
        {
            if (value.Length != 64) return false;
            foreach (var c in value)
                if (!Uri.IsHexDigit(c)) return false;
            return true;
        }

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        private static string Sha256(byte[] bytes)
        {
            using (var hasher = SHA256.Create())
            {
                var hash = hasher.ComputeHash(bytes);
                return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
            }
        }

        private static string SerializeCaptureMetadata(CaptureMetadata metadata)
        {
            var json = JsonUtility.ToJson(metadata);
            if (string.IsNullOrEmpty(metadata.spatial_evidence_sha256))
            {
                // Unity JsonUtility converts a null string to "". The receiver's
                // optional SHA-256 field accepts absent/null, never an empty hash.
                const string emptyField = ",\"spatial_evidence_sha256\":\"\"";
                if (!json.Contains(emptyField))
                    throw new InvalidDataException("Unity metadata serializer changed; capture not committed");
                json = json.Replace(emptyField, "");
            }
            return json;
        }

        private static float[] RowMajor(Matrix4x4 m) => new[]
        {
            m.m00, m.m01, m.m02, m.m03,
            m.m10, m.m11, m.m12, m.m13,
            m.m20, m.m21, m.m22, m.m23,
            m.m30, m.m31, m.m32, m.m33
        };

        [Serializable] private sealed class BridgeConfig
        {
            public string base_url;
            public string wifi_base_url;
            public string token;
            public string certificate_sha256;
            public string project_id;
            public string session_id;
            public string tracking_origin_id;
        }

        [Serializable] private sealed class SessionMarker
        {
            public string project_id, session_id, tracking_origin_id;
        }

        [Serializable] private sealed class Intrinsics
        {
            public float fx, fy, cx, cy;
        }

        [Serializable] private sealed class CaptureMetadata
        {
            public int schema_version = 1;
            public string capture_id, project_id, session_id, tracking_origin_id, camera_id;
            public string image_timestamp_raw, image_timestamp_kind, timestamp_clock;
            public int image_width, image_height;
            public string image_encoding, image_sha256, spatial_evidence_sha256;
            public Intrinsics intrinsics;
            public float[] camera_to_session;
            public string coordinate_convention, position_units;
            public bool tracking_valid;
            public string image_orientation, calibration_revision, capture_source;
        }

        [Serializable] private sealed class SessionInfo
        {
            public string id, project_id, tracking_origin_id, status;
            public int capture_count;
        }

        [Serializable] private sealed class ProjectInfo
        {
            public string id, name;
        }

        private sealed class PinnedCertificateHandler : CertificateHandler
        {
            private readonly string expected;

            public PinnedCertificateHandler(string expectedSha256)
            {
                expected = expectedSha256.ToLowerInvariant();
            }

            protected override bool ValidateCertificate(byte[] certificateData)
            {
                return certificateData != null &&
                       string.Equals(Sha256(certificateData), expected, StringComparison.Ordinal);
            }
        }
    }
}
