using System;
using System.IO;
using System.Security.Cryptography;
using Gsplat;
using UnityEngine;

namespace QuestSplatBridge.Editor
{
    // Read-only diagnostic of the exact runtime loader, independent of anchor/input code.
    // Outputs live only in the caller's evidence directory; never replaces a scan.
    public static class QuestPreviewQualityChecks
    {
        public static void ExportTransport()
        {
            var input = Environment.GetEnvironmentVariable("QUEST_SPLAT_QUALITY_INPUT");
            var output = Environment.GetEnvironmentVariable("QUEST_SPLAT_QUALITY_OUTPUT");
            if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
                throw new InvalidOperationException("Explicit diagnostic input/output required");
            input = Path.GetFullPath(input);
            output = Path.GetFullPath(output);
            if (File.Exists(Path.Combine(output, "transport-receipt.json")))
                throw new InvalidOperationException("Evidence already exists; use another directory");
            var bytes = File.ReadAllBytes(input);
            var hash = Digest(bytes);
            Directory.CreateDirectory(output);
            GsplatAssetUncompressed raw = null;
            GsplatAssetSpark spark = null;
            try
            {
                raw = (GsplatAssetUncompressed)GsplatRuntimeLoader.Load(bytes,
                    GsplatFileFormat.Ply, CompressionMode.Uncompressed, SourceCoordinates.RUB);
                spark = (GsplatAssetSpark)GsplatRuntimeLoader.Load(bytes,
                    GsplatFileFormat.Ply, CompressionMode.Spark, SourceCoordinates.RUB);
                if (raw.SplatCount != spark.SplatCount || raw.SplatCount == 0)
                    throw new InvalidDataException("Loader counts differ");
                using (var writer = new BinaryWriter(File.Create(Path.Combine(output, "uncompressed.bin"))))
                    for (var i = 0; i < raw.SplatCount; i++)
                    {
                        Write(writer, raw.Positions[i]); Write(writer, raw.Scales[i]);
                        Write(writer, raw.Rotations[i]); Write(writer, raw.Colors[i]);
                    }
                using (var writer = new BinaryWriter(File.Create(Path.Combine(output, "spark.bin"))))
                    foreach (var word in spark.PackedSplats)
                    { writer.Write(word.x); writer.Write(word.y); writer.Write(word.z); writer.Write(word.w); }
                if (Digest(File.ReadAllBytes(input)) != hash)
                    throw new InvalidDataException("Input changed during diagnostic");
                var receipt = new TransportReceipt { source = input, sha256 = hash,
                    count = (int)raw.SplatCount, unity_version = Application.unityVersion };
                File.WriteAllText(Path.Combine(output, "transport-receipt.json"), JsonUtility.ToJson(receipt, true));
                Debug.Log("PASS actual runtime transport export: " + raw.SplatCount + " Gaussians; source unchanged");
            }
            finally
            {
                if (raw != null) UnityEngine.Object.DestroyImmediate(raw);
                if (spark != null) UnityEngine.Object.DestroyImmediate(spark);
            }
        }
        private static string Digest(byte[] bytes)
        { using var sha = SHA256.Create(); return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant(); }
        private static void Write(BinaryWriter writer, Vector3 value)
        { writer.Write(value.x); writer.Write(value.y); writer.Write(value.z); }
        private static void Write(BinaryWriter writer, Vector4 value)
        { writer.Write(value.x); writer.Write(value.y); writer.Write(value.z); writer.Write(value.w); }
        [Serializable] private sealed class TransportReceipt
        { public string source, sha256, unity_version; public int count; }
    }
}
