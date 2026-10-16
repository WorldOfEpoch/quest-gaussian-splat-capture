using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace QuestSplatBridge
{
    [Serializable]
    public sealed class CaptureTransferReceipt
    {
        public string capture_id, project_id, session_id, image_sha256, spatial_evidence_sha256, status;
        public bool durable;
    }

    // This pure file/receipt policy is shared by runtime and the failure/retry tests.
    public static class CaptureTransferCleanup
    {
        public static bool Accept(string metadataPath, string captureId, string projectId,
            string sessionId, string sha256, CaptureTransferReceipt receipt)
        {
            return Accept(metadataPath, captureId, projectId, sessionId, sha256, null, receipt);
        }

        public static bool Accept(string metadataPath, string captureId, string projectId,
            string sessionId, string sha256, string spatialSha256, CaptureTransferReceipt receipt)
        {
            if (receipt == null || !receipt.durable || receipt.status != "received" ||
                receipt.capture_id != captureId || receipt.project_id != projectId ||
                receipt.session_id != sessionId || receipt.image_sha256 != sha256 ||
                (!string.IsNullOrEmpty(spatialSha256) && receipt.spatial_evidence_sha256 != spatialSha256))
                return false;
            Guid parsed;
            if (!Guid.TryParse(captureId, out parsed) || !Guid.TryParse(projectId, out parsed) ||
                !Guid.TryParse(sessionId, out parsed) || Path.GetFileNameWithoutExtension(metadataPath) != captureId)
                throw new InvalidDataException("Capture identity invalid");
            var imagePath = Path.ChangeExtension(metadataPath, ".jpg");
            if (!File.Exists(metadataPath) || !File.Exists(imagePath) || Hash(imagePath) != sha256)
                throw new InvalidDataException("Local capture changed; not deleted");
            var sidecar = Path.ChangeExtension(metadataPath, ".spatial.json");
            if (!string.IsNullOrEmpty(spatialSha256) &&
                (!File.Exists(sidecar) || Hash(sidecar) != spatialSha256))
                throw new InvalidDataException("Local spatial evidence changed; not deleted");

            // Persist the matching receipt before deleting either source file. A crash
            // after either deletion resumes from this journal without another upload.
            var journal = Path.ChangeExtension(metadataPath, ".received");
            var temporary = journal + ".tmp";
            var bytes = Encoding.UTF8.GetBytes(captureId + "\n" + projectId + "\n" + sessionId + "\n" + sha256 +
                (string.IsNullOrEmpty(spatialSha256) ? "" : "\n" + spatialSha256));
            using (var stream = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush(true);
            }
            // Existing journals are recovered before retrying this item.
            File.Move(temporary, journal);
            CompleteAcknowledged(journal);
            return true;
        }

        public static void CompleteAcknowledged(string journal)
        {
            if (Path.GetExtension(journal) != ".received") throw new InvalidDataException("Not a receipt journal");
            var values = File.ReadAllText(journal).Split('\n');
            Guid parsed;
            if ((values.Length != 4 && values.Length != 5) || !Guid.TryParse(values[0], out parsed) ||
                !Guid.TryParse(values[1], out parsed) || !Guid.TryParse(values[2], out parsed) ||
                values[0] != Path.GetFileNameWithoutExtension(journal) || values[3].Length != 64 ||
                (values.Length == 5 && values[4].Length != 64))
                throw new InvalidDataException("Receipt journal invalid; files preserved");
            var image = Path.ChangeExtension(journal, ".jpg");
            var spatial = Path.ChangeExtension(journal, ".spatial.json");
            if (File.Exists(image) && Hash(image) != values[3])
                throw new InvalidDataException("Acknowledged image changed; files preserved");
            if (values.Length == 5 && File.Exists(spatial) && Hash(spatial) != values[4])
                throw new InvalidDataException("Acknowledged spatial evidence changed; files preserved");
            File.Delete(image);
            if (values.Length == 5) File.Delete(spatial);
            File.Delete(Path.ChangeExtension(journal, ".json"));
            File.Delete(journal);
        }

        private static string Hash(string path)
        {
            using (var hash = SHA256.Create())
            using (var stream = File.OpenRead(path))
                return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
    }
}
