using System;
using Meta.XR;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace QuestSplatBridge.Editor
{
    /// <summary>Creates our isolated capture scene from Meta's known-working CameraViewer rig.</summary>
    public static class QuestSceneBuilder
    {
        private const string SourceScene =
            "Assets/PassthroughCameraApiSamples/CameraViewer/CameraViewer.unity";
        private const string DestinationScene = "Assets/QuestSplat/QuestCapture.unity";

        [MenuItem("Quest Splat/Create Capture Scene")]
        public static void Build()
        {
            PreserveRuntimeModelShaders();
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(SourceScene) == null)
                throw new InvalidOperationException("Meta CameraViewer scene is absent: " + SourceScene);

            var scene = EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Single);
            var cameraAccess = UnityEngine.Object.FindObjectOfType<PassthroughCameraAccess>(true);
            var mainCamera = Camera.main;
            var rig = GameObject.Find("[BuildingBlock] Camera Rig");
            if (cameraAccess == null || mainCamera == null || rig == null)
                throw new InvalidOperationException("Meta sample rig/camera access is incomplete.");

            // The sample viewer panel is unrelated to our world view and includes its
            // own camera-video plane. Keep the prefab camera-access and passthrough owners.
            foreach (var behaviour in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>(true))
                if (behaviour.GetType().Name == "CameraViewerManager") behaviour.enabled = false;
            // The sample pointer and panel are not part of this controller-only app.
            foreach (var behaviour in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>(true))
                if (behaviour.GetType().Name == "OVRInputModule") behaviour.enabled = false;
            foreach (var behaviour in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>(true))
                if (behaviour != null && behaviour.GetType().Name == "LaserPointer")
                    UnityEngine.Object.DestroyImmediate(behaviour.gameObject);
            foreach (var canvas in UnityEngine.Object.FindObjectsOfType<Canvas>(true))
                canvas.enabled = false;
            // The sample scene may include hand components or visuals independently
            // of our input. Remove them from the copied scene before saving it.
            foreach (var skeleton in UnityEngine.Object.FindObjectsOfType<OVRSkeleton>(true))
                UnityEngine.Object.DestroyImmediate(skeleton);
            foreach (var hand in UnityEngine.Object.FindObjectsOfType<OVRHand>(true))
                UnityEngine.Object.DestroyImmediate(hand);
            // This Meta sample navigation tooltip is placed near the floor and
            // still renders after the sample canvases are disabled.
            foreach (var transform in UnityEngine.Object.FindObjectsOfType<Transform>(true))
            {
                if (transform.name != "ReturnToStartScene") continue;
                UnityEngine.Object.DestroyImmediate(transform.gameObject);
                break;
            }

            var controllerObject = new GameObject("Quest Splat Capture");
            // Runtime PLY loading needs the package's serialized materials/shaders
            // even before any splat is imported into the Editor.
            if (Gsplat.GsplatSettings.Instance == null)
                throw new InvalidOperationException("Gaussian renderer resources are unavailable");
            var controller = controllerObject.AddComponent<QuestCaptureController>();
            var controls = new GameObject("Controller legend (read only)").transform;
            System.IO.Directory.CreateDirectory("Assets/QuestSplat");
            const string backdropPath = "Assets/QuestSplat/ControlsBackdrop.mat";
            var backdropMaterial = AssetDatabase.LoadAssetAtPath<Material>(backdropPath);
            if (backdropMaterial == null)
            {
                var backdropShader = Shader.Find("Quest Splat/HUD backdrop");
                if (backdropShader == null)
                    throw new InvalidOperationException("Control backdrop shader is unavailable");
                backdropMaterial = new Material(backdropShader);
                AssetDatabase.CreateAsset(backdropMaterial, backdropPath);
            }
            var hudShader = Shader.Find("Quest Splat/HUD backdrop");
            if (hudShader == null) throw new InvalidOperationException("HUD backdrop shader is unavailable");
            backdropMaterial.shader = hudShader;
            backdropMaterial.color = new Color(0.015f, 0.035f, 0.055f, 0.88f);
            EditorUtility.SetDirty(backdropMaterial);
            var backdrop = GameObject.CreatePrimitive(PrimitiveType.Quad);
            backdrop.name = "Controls contrast backdrop";
            backdrop.transform.SetParent(controls, false);
            backdrop.transform.localPosition = new Vector3(0, -0.20f, 0.025f);
            backdrop.transform.localRotation = Quaternion.Euler(0, 180, 0);
            backdrop.transform.localScale = new Vector3(0.64f, 0.50f, 1);
            UnityEngine.Object.DestroyImmediate(backdrop.GetComponent<Collider>());
            backdrop.GetComponent<MeshRenderer>().sharedMaterial = backdropMaterial;
            var legendObject = new GameObject("Controller labels");
            legendObject.transform.SetParent(controls, false);
            var legend = legendObject.AddComponent<TextMesh>();
            legend.text = "A Capture   B Splat\nX Mesh      Y Projects\nR Boundary  Menu Exit\nGrip HUD  L click Cycle";
            legend.characterSize = 0.0052f;
            legend.fontSize = 64;
            legend.fontStyle = FontStyle.Bold;
            legend.color = new Color(0.6f, 0.95f, 1f);
            legend.anchor = TextAnchor.UpperCenter;
            legend.alignment = TextAlignment.Center;
            var labelObject = new GameObject("Capture Status");
            labelObject.transform.SetParent(controls, false);
            labelObject.transform.localPosition = new Vector3(0, -0.205f, 0);
            var label = labelObject.AddComponent<TextMesh>();
            label.characterSize = 0.0048f;
            label.fontSize = 64;
            label.fontStyle = FontStyle.Bold;
            label.color = Color.white;
            label.anchor = TextAnchor.UpperCenter;
            label.alignment = TextAlignment.Center;

            var serialized = new SerializedObject(controller);
            const string uncoveredPath = "Assets/QuestSplat/UncoveredPreview.mat";
            var uncovered = AssetDatabase.LoadAssetAtPath<Material>(uncoveredPath);
            if (uncovered == null)
            {
                uncovered = new Material(Shader.Find("Quest Splat/Uncovered view"));
                AssetDatabase.CreateAsset(uncovered, uncoveredPath);
            }
            serialized.FindProperty("uncoveredMaterial").objectReferenceValue = uncovered;
            serialized.FindProperty("cameraAccess").objectReferenceValue = cameraAccess;
            serialized.FindProperty("sessionOrigin").objectReferenceValue = rig.transform;
            serialized.FindProperty("statusCamera").objectReferenceValue = mainCamera;
            serialized.FindProperty("statusText").objectReferenceValue = label;
            serialized.FindProperty("legendPanel").objectReferenceValue = controls;
            serialized.FindProperty("controllerLegendText").objectReferenceValue = legend;
            serialized.ApplyModifiedPropertiesWithoutUndo();

            System.IO.Directory.CreateDirectory("Assets/QuestSplat");
            if (!EditorSceneManager.SaveScene(scene, DestinationScene, true))
                throw new InvalidOperationException("Could not save Quest capture scene.");
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(DestinationScene, true) };
            AssetDatabase.SaveAssets();
            Debug.Log("Quest Splat scene created: " + DestinationScene);
        }

        private static void PreserveRuntimeModelShaders()
        {
            var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset");
            if (assets.Length == 0) throw new InvalidOperationException("Graphics settings missing");
            var settings = new SerializedObject(assets[0]);
            var shaders = settings.FindProperty("m_AlwaysIncludedShaders");
            foreach (var name in new[] { "glTFPbrMetallicRoughness", "glTFPbrSpecularGlossiness", "glTFUnlit" })
            {
                var shader = AssetDatabase.LoadAssetAtPath<Shader>(
                    "Packages/com.unity.cloud.gltfast/Runtime/Shader/Built-In/" + name + ".shader");
                if (shader == null) throw new InvalidOperationException("Runtime model shader missing: " + name);
                var found = false;
                for (var i = 0; i < shaders.arraySize; i++)
                    found |= shaders.GetArrayElementAtIndex(i).objectReferenceValue == shader;
                if (!found)
                {
                    shaders.InsertArrayElementAtIndex(shaders.arraySize);
                    shaders.GetArrayElementAtIndex(shaders.arraySize - 1).objectReferenceValue = shader;
                }
            }
            settings.ApplyModifiedPropertiesWithoutUndo();
            AssetDatabase.SaveAssets();
        }

    }
}
