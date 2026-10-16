using System;
using System.IO;
using System.Text.RegularExpressions;
using System.Xml;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace QuestSplatBridge.Editor
{
    /// <summary>
    /// Unity CLI entrypoint: -batchmode -nographics -buildTarget android
    /// -executeMethod QuestSplatBridge.Editor.QuestBatchBuild.BuildAndroid
    /// </summary>
    public static class QuestBatchBuild
    {
        private const string TestedEditorVersion = "6000.0.66f2";
        private const string ScenePath = "Assets/QuestSplat/QuestCapture.unity";
        private const string AndroidNamespace = "http://schemas.android.com/apk/res/android";

        public static void BuildAndroid()
        {
            if (!string.Equals(Application.unityVersion, TestedEditorVersion, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Quest build requires pinned Unity " + TestedEditorVersion +
                    "; running " + Application.unityVersion);
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                throw new InvalidOperationException(
                    "Launch Unity with -buildTarget android; batch mode cannot switch target here.");

            var applicationId = Environment.GetEnvironmentVariable("QUEST_SPLAT_PACKAGE_ID");
            var metaAppId = Environment.GetEnvironmentVariable("QUEST_SPLAT_META_APP_ID");
            if (string.IsNullOrWhiteSpace(applicationId) ||
                !Regex.IsMatch(applicationId, @"\A[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)+\z"))
                throw new InvalidOperationException("Set QUEST_SPLAT_PACKAGE_ID to your registered Android package ID.");
            if (!ulong.TryParse(metaAppId, out var numericAppId) || numericAppId == 0)
                throw new InvalidOperationException("Set QUEST_SPLAT_META_APP_ID to your own Meta app registration ID.");

            var configuration = Environment.GetEnvironmentVariable("QUEST_SPLAT_BUILD_CONFIGURATION") ?? "development";
            var release = string.Equals(configuration, "release", StringComparison.OrdinalIgnoreCase);
            if (!release && !string.Equals(configuration, "development", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("QUEST_SPLAT_BUILD_CONFIGURATION must be development or release.");
            var releaseKeystore = Environment.GetEnvironmentVariable("QUEST_SPLAT_RELEASE_KEYSTORE");
            var releaseAlias = Environment.GetEnvironmentVariable("QUEST_SPLAT_RELEASE_ALIAS");
            var releasePassword = Environment.GetEnvironmentVariable("QUEST_SPLAT_RELEASE_PASSWORD");
            if (release && (string.IsNullOrWhiteSpace(releaseKeystore) || !File.Exists(releaseKeystore) ||
                string.IsNullOrWhiteSpace(releaseAlias) || string.IsNullOrWhiteSpace(releasePassword)))
                throw new InvalidOperationException("Release builds require the existing keystore, alias and password environment variables.");

            var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            SetControllerOnlyProjectSetting();
            VerifyCameraProjectSetting(projectRoot);
            PrepareManifest(projectRoot);
            // The pinned UnitySplats renderer supports Built-in XR in multi-pass.
            // Preserve the sample's working Built-in passthrough pipeline.
            var xr = UnityEngine.XR.OpenXR.OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
            if (xr == null) throw new InvalidOperationException("Android OpenXR settings are absent");
            xr.renderMode = UnityEngine.XR.OpenXR.OpenXRSettings.RenderMode.MultiPass;
            EditorUtility.SetDirty(xr);
            Oculus.Platform.PlatformSettings.MobileAppID = metaAppId;
            Oculus.Platform.PlatformSettings.UseStandalonePlatform = false;
            Oculus.Platform.PlatformSettings.UseMobileAppIDInEditor = false;
            EditorUtility.SetDirty(Oculus.Platform.PlatformSettings.Instance);
            AssetDatabase.SaveAssets();
            Debug.Log("QuestMetaIdentity: build settings package=" + applicationId + "; standalone=false");
            QuestSceneBuilder.Build();
            QuestHudPlacementChecks.Run();
            QuestMetadataSerializationChecks.Run();

            PlayerSettings.companyName = "Quest Splat";
            PlayerSettings.productName = "Quest Splat Bridge";
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, applicationId);
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            // Historical prototype targets, retained for source parity. Check current
            // platform requirements before publishing a new store/channel build.
            PlayerSettings.Android.targetSdkVersion = release
                ? AndroidSdkVersions.AndroidApiLevel34
                : AndroidSdkVersions.AndroidApiLevel35;
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
            EditorUserBuildSettings.buildAppBundle = false;
            PlayerSettings.Android.bundleVersionCode = Math.Max(PlayerSettings.Android.bundleVersionCode + 1, 20261002);

            var requestedApk = Environment.GetEnvironmentVariable("QUEST_SPLAT_APK");
            var apk = string.IsNullOrWhiteSpace(requestedApk)
                ? Path.Combine(projectRoot, "Builds", "QuestSplat.apk")
                : Path.GetFullPath(Path.IsPathRooted(requestedApk)
                    ? requestedApk
                    : Path.Combine(projectRoot, requestedApk));
            if (!string.Equals(Path.GetExtension(apk), ".apk", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("QUEST_SPLAT_APK must name an .apk file.");
            Directory.CreateDirectory(Path.GetDirectoryName(apk) ?? projectRoot);
            if (File.Exists(apk)) File.Delete(apk);

            var oldCustomKeystore = PlayerSettings.Android.useCustomKeystore;
            var oldKeystore = PlayerSettings.Android.keystoreName;
            var oldAlias = PlayerSettings.Android.keyaliasName;
            var oldKeystorePassword = PlayerSettings.Android.keystorePass;
            var oldAliasPassword = PlayerSettings.Android.keyaliasPass;
            try
            {
                if (release)
                {
                    PlayerSettings.Android.useCustomKeystore = true;
                    PlayerSettings.Android.keystoreName = Path.GetFullPath(releaseKeystore);
                    PlayerSettings.Android.keyaliasName = releaseAlias;
                    PlayerSettings.Android.keystorePass = releasePassword;
                    PlayerSettings.Android.keyaliasPass = releasePassword;
                }
                AssetDatabase.SaveAssets();
                var options = new BuildPlayerOptions
                {
                    scenes = new[] { ScenePath },
                    locationPathName = apk,
                    target = BuildTarget.Android,
                    targetGroup = BuildTargetGroup.Android,
                    options = release ? BuildOptions.None : BuildOptions.Development
                };
                var report = BuildPipeline.BuildPlayer(options);
                if (report == null || report.summary.result != BuildResult.Succeeded ||
                    !File.Exists(apk) ||
                    new FileInfo(apk).Length == 0)
                {
                    var result = report == null ? "no BuildReport" : report.summary.result.ToString();
                    var errors = report == null ? "unavailable" : report.summary.totalErrors.ToString();
                    throw new InvalidOperationException(
                        "Quest APK build failed: " + result + ", errors=" + errors + ", output=" + apk);
                }
                if (report.summary.totalErrors > 0)
                    Debug.LogWarning("Unity reported " + report.summary.totalErrors +
                        " nonfatal error(s) despite BuildResult.Succeeded; review the build log.");
                Debug.Log("Quest APK built (" + configuration + "): " + apk + " (" + new FileInfo(apk).Length + " bytes)");
            }
            finally
            {
                // Signing material stays local; preparing a channel build must not
                // silently change the signer used by later data-preserving USB updates.
                PlayerSettings.Android.useCustomKeystore = oldCustomKeystore;
                PlayerSettings.Android.keystoreName = oldKeystore;
                PlayerSettings.Android.keyaliasName = oldAlias;
                PlayerSettings.Android.keystorePass = oldKeystorePassword;
                PlayerSettings.Android.keyaliasPass = oldAliasPassword;
                AssetDatabase.SaveAssets();
            }
        }

        private static void SetControllerOnlyProjectSetting()
        {
            const string path = "Assets/Oculus/OculusProjectConfig.asset";
            var config = AssetDatabase.LoadAssetAtPath<OVRProjectConfig>(path);
            if (config == null)
                throw new InvalidOperationException("Meta Quest project config is missing: " + path);
            config.handTrackingSupport = OVRProjectConfig.HandTrackingSupport.ControllersOnly;
            // Contextual boundary suppression is available only when this
            // capability is declared for the standalone Quest APK.
            config.boundaryVisibilitySupport = OVRProjectConfig.FeatureSupport.Supported;
            EditorUtility.SetDirty(config);
            AssetDatabase.SaveAssets();
        }

        private static void VerifyCameraProjectSetting(string projectRoot)
        {
            var configPath = Path.Combine(projectRoot, "Assets", "Oculus", "OculusProjectConfig.asset");
            if (!File.Exists(configPath) ||
                !File.ReadAllText(configPath).Contains("isPassthroughCameraAccessEnabled: 1"))
                throw new InvalidOperationException(
                    "Meta sample does not have Passthrough Camera Access enabled: " + configPath);
            if (!File.ReadAllText(configPath).Contains("handTrackingSupport: 0"))
                throw new InvalidOperationException("Quest Splat requires controller-only input.");
            if (!File.ReadAllText(configPath).Contains("boundaryVisibilitySupport: 1"))
                throw new InvalidOperationException("Quest Splat requires contextual boundary visibility support.");
        }

        private static void PrepareManifest(string projectRoot)
        {
            var path = Path.Combine(projectRoot, "Assets", "Plugins", "Android", "AndroidManifest.xml");
            if (!File.Exists(path))
                throw new InvalidOperationException("Meta sample AndroidManifest.xml is missing.");
            var document = new XmlDocument { PreserveWhitespace = true };
            document.Load(path);
            var root = document.DocumentElement;
            var application = root?.SelectSingleNode("application") as XmlElement;
            if (root == null || application == null)
                throw new InvalidOperationException("Meta Android manifest is malformed.");

            var cameraPermissionPresent = false;
            var internetPermissionPresent = false;
            var boundaryPermissionPresent = false;
            for (var i = root.ChildNodes.Count - 1; i >= 0; --i)
            {
                var element = root.ChildNodes[i] as XmlElement;
                if (element == null) continue;
                var name = element.GetAttribute("name", AndroidNamespace);
                if ((element.LocalName == "uses-permission" || element.LocalName == "uses-feature") &&
                    (name == "com.oculus.permission.HAND_TRACKING" ||
                     name == "oculus.software.handtracking" ||
                     name == "com.oculus.feature.HAND_TRACKING"))
                {
                    root.RemoveChild(element);
                    continue;
                }
                if (element.LocalName != "uses-permission") continue;
                if (name == "horizonos.permission.HEADSET_CAMERA") cameraPermissionPresent = true;
                if (name == "android.permission.INTERNET") internetPermissionPresent = true;
                if (name == "com.oculus.permission.BOUNDARY_VISIBILITY") boundaryPermissionPresent = true;
            }
            if (!cameraPermissionPresent)
                throw new InvalidOperationException("The physical headset camera permission is absent.");
            if (!internetPermissionPresent)
            {
                var permission = document.CreateElement("uses-permission");
                permission.SetAttribute("name", AndroidNamespace, "android.permission.INTERNET");
                root.AppendChild(permission);
            }
            if (!boundaryPermissionPresent)
            {
                var permission = document.CreateElement("uses-permission");
                permission.SetAttribute("name", AndroidNamespace, "com.oculus.permission.BOUNDARY_VISIBILITY");
                root.AppendChild(permission);
            }

            // Unity's launcher supplies the app label from PlayerSettings.productName.
            // A label here conflicts with the launcher during manifest merging.
            application.RemoveAttribute("label", AndroidNamespace);
            foreach (XmlNode node in application.SelectNodes("activity"))
            {
                var activity = node as XmlElement;
                if (activity?.SelectSingleNode("intent-filter/action[@android:name='android.intent.action.MAIN']",
                    CreateAndroidNamespaceManager(document)) != null)
                    activity.SetAttribute("screenOrientation", AndroidNamespace, "landscape");
            }
            for (var i = application.ChildNodes.Count - 1; i >= 0; --i)
            {
                var element = application.ChildNodes[i] as XmlElement;
                if (element != null &&
                    element.LocalName == "meta-data" &&
                    element.GetAttribute("name", AndroidNamespace) == "com.oculus.telemetry.project_guid")
                    application.RemoveChild(element);
            }
            document.Save(path);
            AssetDatabase.ImportAsset("Assets/Plugins/Android/AndroidManifest.xml",
                ImportAssetOptions.ForceSynchronousImport);
        }

        private static XmlNamespaceManager CreateAndroidNamespaceManager(XmlDocument document)
        {
            var manager = new XmlNamespaceManager(document.NameTable);
            manager.AddNamespace("android", AndroidNamespace);
            return manager;
        }
    }
}
