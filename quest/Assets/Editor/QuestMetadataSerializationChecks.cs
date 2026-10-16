using System;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace QuestSplatBridge.Editor
{
    public static class QuestMetadataSerializationChecks
    {
        public static void Run()
        {
            var metadataType = typeof(QuestCaptureController).GetNestedType("CaptureMetadata", BindingFlags.NonPublic);
            if (metadataType == null) throw new InvalidOperationException("Capture metadata owner missing");
            var metadata = Activator.CreateInstance(metadataType, true);
            var hashField = metadataType.GetField("spatial_evidence_sha256", BindingFlags.Instance | BindingFlags.Public);
            if (hashField == null) throw new InvalidOperationException("Spatial hash field missing");

            hashField.SetValue(metadata, null);
            var serializer = typeof(QuestCaptureController).GetMethod("SerializeCaptureMetadata",
                BindingFlags.NonPublic | BindingFlags.Static);
            if (serializer == null) throw new InvalidOperationException("Capture metadata serializer missing");
            var absent = (string)serializer.Invoke(null, new[] { metadata });
            if (absent.Contains("\"spatial_evidence_sha256\""))
                throw new InvalidOperationException("No-sidecar metadata still contains spatial hash field");
            Debug.Log("Quest metadata no-sidecar serialization omitted hash field");

            hashField.SetValue(metadata, new string('a', 64));
            var present = (string)serializer.Invoke(null, new[] { metadata });
            if (!present.Contains("\"spatial_evidence_sha256\":\"" + new string('a', 64) + "\""))
                throw new InvalidOperationException("Present spatial hash failed Unity serialization");
            Debug.Log("Quest metadata present spatial hash serialization passed");
            CheckDenseSpatialEnvelope();
            CheckDepthPreviewFixture();
            CheckPreviewRevisionGate();
        }

        private static void CheckPreviewRevisionGate()
        {
            var owner = typeof(QuestCaptureController);
            var flags = BindingFlags.NonPublic | BindingFlags.Instance;
            var manifestType = owner.GetNestedType("PreviewManifest", BindingFlags.NonPublic);
            var stale = owner.GetMethod("IsStalePreviewRevision", flags);
            var observe = owner.GetMethod("ObservePreviewRevision", flags);
            var dispose = owner.GetMethod("DisposePreview", flags);
            if (manifestType == null || stale == null || observe == null || dispose == null)
                throw new InvalidOperationException("Preview revision owner missing");
            var revisionField = manifestType.GetField("revision_number");
            var representationField = manifestType.GetField("representation");
            if (revisionField == null || representationField == null)
                throw new InvalidOperationException("Preview revision fields missing");
            var parsed = JsonUtility.FromJson("{\"representation\":\"depth\",\"revision_number\":3000000000}", manifestType);
            if ((long)revisionField.GetValue(parsed) != 3000000000L)
                throw new InvalidOperationException("Unity JSON lost the ordered long revision");
            var root = new GameObject("Preview revision check");
            try
            {
                var controller = root.AddComponent<QuestCaptureController>();
                var depth = Activator.CreateInstance(manifestType, true);
                representationField.SetValue(depth, "depth");
                bool IsStale(object item) => (bool)stale.Invoke(controller, new[] { item });
                if (IsStale(depth)) throw new InvalidOperationException("Legacy depth was rejected before a numbered revision");
                revisionField.SetValue(depth, 7L);
                observe.Invoke(controller, new[] { depth });
                revisionField.SetValue(depth, 0L);
                if (!IsStale(depth)) throw new InvalidOperationException("Legacy depth replaced a numbered revision");
                revisionField.SetValue(depth, 6L);
                if (!IsStale(depth)) throw new InvalidOperationException("Older depth revision was accepted");
                revisionField.SetValue(depth, 7L);
                if (IsStale(depth)) throw new InvalidOperationException("Current depth revision was rejected");
                var splat = Activator.CreateInstance(manifestType, true);
                representationField.SetValue(splat, "splat");
                if (IsStale(splat)) throw new InvalidOperationException("Depth revision incorrectly gated splat");
                dispose.Invoke(controller, new object[] { false });
                revisionField.SetValue(depth, 6L);
                if (!IsStale(depth)) throw new InvalidOperationException("Preview replacement reset the revision gate");
                dispose.Invoke(controller, new object[] { true });
                if (IsStale(depth)) throw new InvalidOperationException("Session preview reset retained an old revision gate");
            }
            finally { UnityEngine.Object.DestroyImmediate(root); }
            Debug.Log("Quest ordered preview revision gate passed");
        }

        private static void CheckDepthPreviewFixture()
        {
            var path = Environment.GetEnvironmentVariable("QUEST_SPLAT_DEPTH_PREVIEW_FIXTURE");
            if (string.IsNullOrEmpty(path)) return;
            var asset = Gsplat.GsplatRuntimeLoader.Load(System.IO.File.ReadAllBytes(path),
                Gsplat.GsplatFileFormat.Ply, Gsplat.CompressionMode.Spark, Gsplat.SourceCoordinates.RUB);
            try
            {
                if (asset.SplatCount <= 0 || asset.SplatCount > 160000 ||
                    float.IsNaN(asset.Bounds.center.x) || float.IsNaN(asset.Bounds.center.y) ||
                    float.IsNaN(asset.Bounds.center.z))
                    throw new InvalidOperationException("Observed depth fixture did not load within its display budget");
                Debug.Log("Quest depth fixture loaded: " + asset.SplatCount + "; bounds=" + asset.Bounds);
            }
            finally { UnityEngine.Object.DestroyImmediate(asset); }
        }

        private static void CheckDenseSpatialEnvelope()
        {
            var owner = typeof(QuestCaptureController);
            var flags = BindingFlags.NonPublic | BindingFlags.Static;
            var columns = (int)owner.GetField("SpatialColumns", flags).GetRawConstantValue();
            var rows = (int)owner.GetField("SpatialRows", flags).GetRawConstantValue();
            var stride = (int)owner.GetField("SpatialPermutationStride", flags).GetRawConstantValue();
            var timeBudget = (double)owner.GetField("SpatialTimeBudgetSeconds", flags).GetRawConstantValue();
            var count = columns * rows;
            if (columns != 48 || rows != 32 || count > 4096 || timeBudget > .1 || timeBudget <= 0)
                throw new InvalidOperationException("Dense survey exceeds its ray or time envelope");
            var visited = new bool[count];
            for (var i = 0; i < count; i++)
            {
                var index = (i * stride) % count;
                if (visited[index]) throw new InvalidOperationException("Dense grid permutation repeats a ray");
                visited[index] = true;
            }

            var evidenceType = owner.GetNestedType("SpatialEvidence", BindingFlags.NonPublic);
            var rayType = owner.GetNestedType("SpatialRay", BindingFlags.NonPublic);
            var serializer = owner.GetMethod("SerializeSpatial", flags);
            if (evidenceType == null || rayType == null || serializer == null)
                throw new InvalidOperationException("Spatial evidence serializer missing");
            var evidence = Activator.CreateInstance(evidenceType, true);
            foreach (var name in new[] { "capture_id", "project_id", "session_id", "tracking_origin_id" })
                evidenceType.GetField(name).SetValue(evidence, Guid.NewGuid().ToString());
            evidenceType.GetField("rgb_timestamp_raw").SetValue(evidence, "2026-10-12T15:00:00.0000000Z");
            evidenceType.GetField("rgb_timestamp_kind").SetValue(evidence, "Utc");
            evidenceType.GetField("availability").SetValue(evidence, "ready");
            evidenceType.GetField("trigger_realtime_s").SetValue(evidence, 1.0);
            evidenceType.GetField("sample_started_realtime_s").SetValue(evidence, 1.01);
            evidenceType.GetField("sample_completed_realtime_s").SetValue(evidence, 1.09);
            var rays = Array.CreateInstance(rayType, count);
            for (var i = 0; i < count; i++)
            {
                var ray = Activator.CreateInstance(rayType, true);
                rayType.GetField("origin").SetValue(ray, new[] { 0f, 0f, 0f });
                rayType.GetField("direction").SetValue(ray, new[] { 0f, 0f, 1f });
                rayType.GetField("status").SetValue(ray, "no_hit");
                rays.SetValue(ray, i);
            }
            evidenceType.GetField("rays").SetValue(evidence, rays);
            var bytes = (byte[])serializer.Invoke(null, new[] { evidence });
            if (bytes.Length > 2 * 1024 * 1024)
                throw new InvalidOperationException("Dense survey exceeds the receiver's 2 MiB limit");
            var json = Encoding.UTF8.GetString(bytes);
            if (json.Contains("\"normal_confidence\"") || json.Contains("\"hit\""))
                throw new InvalidOperationException("No-hit rays leaked hit-only fields");
            var decoded = JsonUtility.FromJson(json, evidenceType);
            if (((Array)evidenceType.GetField("rays").GetValue(decoded)).Length != count)
                throw new InvalidOperationException("Dense survey serialization lost rays");
            var outputPath = Environment.GetEnvironmentVariable("QUEST_SPATIAL_TEST_JSON");
            if (!string.IsNullOrEmpty(outputPath)) System.IO.File.WriteAllBytes(outputPath, bytes);
            Debug.Log("Quest dense spatial survey envelope passed: " + count + " rays, " + bytes.Length + " bytes");
        }
    }
}
