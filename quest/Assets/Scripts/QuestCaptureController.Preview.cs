using System;
using System.Collections;
using System.IO;
using Gsplat;
using UnityEngine;
using UnityEngine.Networking;

namespace QuestSplatBridge
{
    public sealed partial class QuestCaptureController
    {
        private GameObject previewObject;
        private GsplatAsset previewAsset;
        private string previewHash;
        private string previewRepresentation;
        private bool IsPreviewSelected => previewObject != null && requestedPreview == previewRepresentation;
        private static bool IsRenderedPreview(string representation) => representation == "splat" || representation == "depth";
        private bool previewAligned;
        private int previewAcceptedViews;
        private int previewTotalViews;
        private long highestDepthRevision;
        private long highestSplatRevision;
        [SerializeField] private Material uncoveredMaterial;
        private GameObject uncoveredView;
        private const long MaxPreviewBytes = 64L * 1024L * 1024L;
        private void Preview(string representation)
        {
            if (requestedPreview == representation)
            {
                requestedPreview = null;
                if (previewObject != null) previewObject.SetActive(false);
                UpdateCoverageTint();
                status = "Passthrough; capture remains active";
                return;
            }
            requestedPreview = representation;
            if (IsRenderedPreview(representation) && previewObject != null && hudMode == HudDisplayMode.Full)
            {
                SetHudMode(HudDisplayMode.Compact);
            }
            else if (IsRenderedPreview(representation) && previewObject == null) ShowFullMenu();
            if (previewObject != null) previewObject.SetActive(IsPreviewSelected && previewAligned && AnchorReady);
            status = IsPreviewSelected ? PreviewLabel() : representation == "depth" ? "Loading shared depth map" :
                representation == "splat" ? "Loading latest splat" : "Mesh has not been generated";
            UpdateCoverageTint();
        }

        private string PreviewLabel() => previewRepresentation == "depth"
            ? "Depth samples: " + previewAcceptedViews + " views; provisional"
            : previewAligned ? "Splat overlay: " + previewAcceptedViews + " views"
                : "Splat model; physical alignment unavailable";

        private IEnumerator PreviewRefreshLoop()
        {
            while (true)
            {
                if (Configured() && projectVerified)
                    yield return CheckPreview();
                yield return new WaitForSecondsRealtime(3);
            }
        }

        private void UpdateCoverageTint()
        {
            var show = previewAligned && AnchorReady &&
                requestedPreview == "splat" && IsPreviewSelected;
            if (show && uncoveredView == null && uncoveredMaterial != null)
            {
                uncoveredView = GameObject.CreatePrimitive(PrimitiveType.Quad);
                uncoveredView.name = "Uncovered preview tint";
                Destroy(uncoveredView.GetComponent<Collider>());
                uncoveredView.transform.SetParent(statusCamera.transform, false);
                uncoveredView.transform.localPosition = new Vector3(0, 0, 0.5f);
                uncoveredView.transform.localScale = Vector3.one * 4;
                uncoveredView.GetComponent<MeshRenderer>().sharedMaterial = uncoveredMaterial;
            }
            if (uncoveredView != null) uncoveredView.SetActive(show);
        }

        private IEnumerator CheckPreview()
        {
            var generation = configGeneration;
            var representation = requestedPreview == "depth" ? "depth" : "splat";
            foreach (var endpoint in Endpoints())
            {
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + "/api/v1/sessions/" +
                    UnityWebRequest.EscapeURL(config.session_id) + "/preview?representation=" + representation))
                {
                    ConfigureCertificate(request);
                    request.timeout = 5;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (generation != configGeneration) yield break;
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (request.result != UnityWebRequest.Result.Success)
                    { if (!IsPreviewSelected) status = "Preview request failed: " + request.responseCode; yield break; }
                    PreviewManifest manifest = null;
                    try { manifest = JsonUtility.FromJson<PreviewManifest>(request.downloadHandler.text); }
                    catch (Exception ex) { Debug.LogException(ex); }
                    // Poll counts in camera mode; load only the explicitly selected view.
                    if (manifest == null || IsStalePreviewRevision(manifest)) yield break;
                    if (requestedPreview != representation)
                    {
                        previewTotalViews = Mathf.Max(previewTotalViews, manifest.total_views);
                        yield break;
                    }
                    if (manifest.status != "ready" || manifest.artifact == null)
                    {
                        previewTotalViews = Mathf.Max(previewTotalViews, manifest.total_views);
                        if (!IsPreviewSelected)
                        {
                            status = manifest?.reason ?? "No preview ready on PC";
                        }
                        yield break;
                    }
                    if (manifest.representation != representation || manifest.project_id != config.project_id ||
                        manifest.session_id != config.session_id || !ValidSha256(manifest.artifact.sha256) ||
                        manifest.artifact.size_bytes <= 0 || manifest.artifact.size_bytes > MaxPreviewBytes ||
                        manifest.artifact.vertex_count <= 0 || manifest.artifact.vertex_count > 250000 ||
                        manifest.artifact.download_path == null ||
                        !manifest.artifact.download_path.StartsWith("/api/v1/projects/" + config.project_id + "/", StringComparison.Ordinal) ||
                        manifest.artifact.download_path.Contains("..") || manifest.artifact.download_path.Contains("?"))
                    { status = "Preview manifest rejected"; yield break; }
                    var locationEligible = manifest.alignment != null && manifest.alignment.headset_preview_allowed &&
                        manifest.session_id == config.session_id && manifest.tracking_origin_id == config.tracking_origin_id;
                    if (!locationEligible)
                    { status = "Scan alignment unavailable; keeping room overlay"; yield break; }
                    ObservePreviewRevision(manifest);
                    previewTotalViews = Mathf.Max(previewTotalViews, manifest.total_views);
                    if (locationEligible && !AnchorReady)
                    { status = "Locating scan before overlay; look around"; yield break; }
                    if (manifest.artifact.sha256 == previewHash && previewRepresentation == representation && previewAligned)
                    {
                        previewAcceptedViews = Mathf.Max(0, manifest.accepted_views);
                        yield break;
                    }
                    yield return DownloadPreview(endpoint, manifest, generation);
                    yield break;
                }
            }
        }

        private IEnumerator DownloadPreview(string endpoint, PreviewManifest manifest, int generation)
        {
            if (IsStalePreviewRevision(manifest)) yield break;
            var artifact = manifest.artifact;
            using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + artifact.download_path))
            {
                ConfigureCertificate(request);
                request.timeout = 45;
                request.SetRequestHeader("Authorization", "Bearer " + config.token);
                var operation = request.SendWebRequest();
                while (!operation.isDone)
                {
                    if (request.downloadedBytes > (ulong)artifact.size_bytes ||
                        requestedPreview != manifest.representation || generation != configGeneration)
                    { request.Abort(); yield break; }
                    yield return null;
                }
                if (generation != configGeneration || IsStalePreviewRevision(manifest)) yield break;
                if (request.result != UnityWebRequest.Result.Success)
                { status = "Preview transfer interrupted; keeping previous view"; yield break; }
                var bytes = request.downloadHandler.data;
                if (bytes.LongLength != artifact.size_bytes || Sha256(bytes) != artifact.sha256)
                { status = "Preview checksum mismatch; keeping previous view"; yield break; }
                GameObject candidate = null;
                GsplatAsset asset = null;
                bool aligned = false;
                try
                {
                    asset = GsplatRuntimeLoader.Load(bytes, GsplatFileFormat.Ply, CompressionMode.Spark, SourceCoordinates.RUB);
                    if (asset.SplatCount != artifact.vertex_count) throw new InvalidDataException("Splat count mismatch");
                    candidate = new GameObject((manifest.representation == "depth" ? "Observed depth " : "Gaussian preview ") + manifest.job_id);
                    var renderer = candidate.AddComponent<GsplatRenderer>();
                    renderer.AsyncUpload = true;
                    renderer.RenderBeforeUploadComplete = false;
                    // Keep the trained model's view-dependent detail. Depth markers have no SH bands.
                    renderer.SHDegree = manifest.representation == "depth" ? 0 : 3;
                    renderer.GsplatAsset = asset;
                    var a = manifest.alignment;
                    aligned = a != null && a.headset_preview_allowed && manifest.session_id == config.session_id &&
                        manifest.tracking_origin_id == config.tracking_origin_id && AnchorReady;
                    if (!aligned)
                        throw new InvalidDataException("Room overlay requires its located session anchor");
                    ApplyRoomPreviewTransform(candidate.transform, a.unity_from_splat);
                }
                catch (Exception error)
                {
                    if (candidate != null) Destroy(candidate);
                    if (asset != null) Destroy(asset);
                    candidate = null;
                    status = "Preview load failed; keeping previous view";
                    Debug.LogException(error);
                }
                if (candidate == null) yield break;
                var readyBy = Time.unscaledTime + 15;
                var candidateRenderer = candidate.GetComponent<GsplatRenderer>();
                while (!candidateRenderer.Valid && Time.unscaledTime < readyBy &&
                       requestedPreview == manifest.representation && generation == configGeneration) yield return null;
                if (!candidateRenderer.Valid || requestedPreview != manifest.representation ||
                    generation != configGeneration || IsStalePreviewRevision(manifest))
                { Destroy(candidate); Destroy(asset); yield break; }
                var replacing = previewObject != null;
                var lastTotalViews = previewTotalViews;
                DisposePreview(false);
                previewObject = candidate;
                previewAsset = asset;
                previewHash = artifact.sha256;
                previewRepresentation = manifest.representation;
                previewAligned = aligned;
                previewAcceptedViews = manifest.accepted_views;
                previewTotalViews = Mathf.Max(lastTotalViews, manifest.total_views);
                previewObject.SetActive(IsPreviewSelected && previewAligned && AnchorReady);
                if (!IsLibraryOpen && hudMode == HudDisplayMode.Full) SetHudMode(HudDisplayMode.Compact);
                UpdateCoverageTint();
                status = replacing && aligned && previewRepresentation == "splat"
                    ? "Splat updated: " + previewAcceptedViews + " views"
                    : PreviewLabel();
                Debug.Log("Quest Splat: rendered " + artifact.vertex_count + "; representation=" + previewRepresentation + "; aligned=" + aligned);
            }
        }

        private void ApplyRoomPreviewTransform(Transform target, float[] unityFromSplat)
        {
            if (sessionOrigin == null) throw new InvalidDataException("Missing scan anchor");
            var matrix = Matrix(unityFromSplat);
            // Room coordinates belong to the persisted anchor. Neither the current
            // headset view nor the model bounds may recenter or move this overlay.
            target.SetParent(sessionOrigin, false);
            target.localPosition = matrix.GetColumn(3);
            target.localRotation = matrix.rotation;
            target.localScale = matrix.lossyScale;
        }

        private static Matrix4x4 Matrix(float[] rowMajor)
        {
            if (rowMajor == null || rowMajor.Length != 16) throw new InvalidDataException("Missing preview transform");
            var result = new Matrix4x4();
            for (var r = 0; r < 4; r++) for (var c = 0; c < 4; c++)
            {
                var v = rowMajor[r * 4 + c];
                if (!Finite(v)) throw new InvalidDataException("Non-finite preview transform");
                result[r, c] = v;
            }
            if (result.determinant <= 0) throw new InvalidDataException("Invalid preview handedness");
            return result;
        }

        private bool IsStalePreviewRevision(PreviewManifest manifest)
        {
            if (manifest.revision_number < 0) return true;
            var highest = manifest.representation == "depth" ? highestDepthRevision : highestSplatRevision;
            return highest > 0 && manifest.revision_number < highest;
        }

        private void ObservePreviewRevision(PreviewManifest manifest)
        {
            if (manifest.representation == "depth")
                highestDepthRevision = Math.Max(highestDepthRevision, manifest.revision_number);
            else if (manifest.representation == "splat")
                highestSplatRevision = Math.Max(highestSplatRevision, manifest.revision_number);
        }

        private void DisposePreview(bool resetRevision = true)
        {
            if (previewObject != null) Destroy(previewObject);
            if (previewAsset != null) Destroy(previewAsset);
            previewObject = null;
            previewAsset = null;
            previewHash = null;
            previewRepresentation = null;
            previewAligned = false;
            previewAcceptedViews = 0;
            previewTotalViews = 0;
            if (resetRevision)
            {
                highestDepthRevision = 0;
                highestSplatRevision = 0;
            }
            if (uncoveredView != null) uncoveredView.SetActive(false);
        }

        [Serializable] private sealed class PreviewManifest
        {
            public string status, reason, project_id, session_id, tracking_origin_id, job_id, representation;
            public int accepted_views, total_views;
            public long revision_number;
            public PreviewArtifact artifact;
            public PreviewAlignment alignment;
        }
        [Serializable] private sealed class PreviewArtifact
        {
            public string sha256, download_path;
            public long size_bytes;
            public int vertex_count;
        }
        [Serializable] private sealed class PreviewAlignment
        {
            public bool headset_preview_allowed;
            public float[] unity_from_splat;
        }
    }
}
