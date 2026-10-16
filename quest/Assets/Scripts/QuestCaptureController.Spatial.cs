using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text;
using Meta.XR;
using UnityEngine;

namespace QuestSplatBridge
{
    public sealed partial class QuestCaptureController
    {
        private EnvironmentRaycastManager spatialRaycaster;
        private GameObject ownedSpatialRaycaster;
        private bool spatialWarmupLogged;
        private const int SpatialColumns = 48;
        private const int SpatialRows = 32;
        // Coprime with 48 * 32, so a budget-limited prefix covers the view.
        private const int SpatialPermutationStride = 997;
        private const double SpatialTimeBudgetSeconds = 0.10;

        [Serializable] private sealed class SpatialEvidence
        {
            public int schema_version = 1;
            public string capture_id, project_id, session_id, tracking_origin_id;
            public string source = "meta_environment_raycast_openxr";
            public string association = "near_capture_unaligned";
            public string coordinate_frame = "quest_session";
            public string coordinate_convention = "unity_lh_xright_yup_zforward";
            public string position_units = "meters";
            public double trigger_realtime_s, sample_started_realtime_s, sample_completed_realtime_s;
            public string rgb_timestamp_raw, rgb_timestamp_kind, availability;
            public SpatialRay[] rays = Array.Empty<SpatialRay>();
        }

        [Serializable] private sealed class SpatialRay
        {
            public float[] origin, direction;
            public string status;
            public float[] hit, normal;
            public float normal_confidence;
        }

        private static string SpatialPath(string metadataPath) =>
            Path.ChangeExtension(metadataPath, ".spatial.json");

        private static string[] CaptureMetadataPaths(string folder)
        {
            var candidates = Directory.GetFiles(folder, "*.json");
            return Array.FindAll(candidates, path => !path.EndsWith(".spatial.json", StringComparison.OrdinalIgnoreCase));
        }

        private static float[] Triple(Vector3 value) => new[] { value.x, value.y, value.z };

        private static void JsonText(StringBuilder output, string value)
        {
            output.Append('"');
            foreach (var character in value ?? "")
            {
                switch (character)
                {
                    case '"': output.Append("\\\""); break;
                    case '\\': output.Append("\\\\"); break;
                    case '\n': output.Append("\\n"); break;
                    case '\r': output.Append("\\r"); break;
                    case '\t': output.Append("\\t"); break;
                    default:
                        if (character < 32) output.Append("\\u").Append(((int)character).ToString("x4"));
                        else output.Append(character);
                        break;
                }
            }
            output.Append('"');
        }

        private static void JsonField(StringBuilder output, string name, string value)
        { JsonText(output, name); output.Append(':'); JsonText(output, value); }

        private static void JsonNumber(StringBuilder output, double value) =>
            output.Append(value.ToString("R", CultureInfo.InvariantCulture));

        private static void JsonTriple(StringBuilder output, float[] values)
        {
            output.Append('[');
            for (var i = 0; i < 3; i++)
            {
                if (i > 0) output.Append(',');
                JsonNumber(output, values[i]);
            }
            output.Append(']');
        }

        private static byte[] SerializeSpatial(SpatialEvidence evidence)
        {
            var output = new StringBuilder(32768);
            output.Append("{\"schema_version\":1,");
            JsonField(output, "capture_id", evidence.capture_id); output.Append(',');
            JsonField(output, "project_id", evidence.project_id); output.Append(',');
            JsonField(output, "session_id", evidence.session_id); output.Append(',');
            JsonField(output, "tracking_origin_id", evidence.tracking_origin_id); output.Append(',');
            JsonField(output, "source", evidence.source); output.Append(',');
            JsonField(output, "association", evidence.association); output.Append(',');
            JsonField(output, "coordinate_frame", evidence.coordinate_frame); output.Append(',');
            JsonField(output, "coordinate_convention", evidence.coordinate_convention); output.Append(',');
            JsonField(output, "position_units", evidence.position_units); output.Append(',');
            output.Append("\"trigger_realtime_s\":"); JsonNumber(output, evidence.trigger_realtime_s); output.Append(',');
            output.Append("\"sample_started_realtime_s\":"); JsonNumber(output, evidence.sample_started_realtime_s); output.Append(',');
            output.Append("\"sample_completed_realtime_s\":"); JsonNumber(output, evidence.sample_completed_realtime_s); output.Append(',');
            JsonField(output, "rgb_timestamp_raw", evidence.rgb_timestamp_raw); output.Append(',');
            JsonField(output, "rgb_timestamp_kind", evidence.rgb_timestamp_kind); output.Append(',');
            JsonField(output, "availability", evidence.availability);
            output.Append(",\"rays\":[");
            for (var i = 0; i < evidence.rays.Length; i++)
            {
                var ray = evidence.rays[i];
                if (i > 0) output.Append(',');
                output.Append("{\"origin\":"); JsonTriple(output, ray.origin);
                output.Append(",\"direction\":"); JsonTriple(output, ray.direction);
                output.Append(','); JsonField(output, "status", ray.status);
                if (ray.status == "hit")
                {
                    output.Append(",\"hit\":"); JsonTriple(output, ray.hit);
                    output.Append(",\"normal\":"); JsonTriple(output, ray.normal);
                    output.Append(",\"normal_confidence\":"); JsonNumber(output, ray.normal_confidence);
                }
                output.Append('}');
            }
            output.Append("]}");
            return Encoding.UTF8.GetBytes(output.ToString());
        }

        private void DisposeSpatialRaycaster()
        {
            if (ownedSpatialRaycaster != null) Destroy(ownedSpatialRaycaster);
            ownedSpatialRaycaster = null;
            spatialRaycaster = null;
        }

        private bool EnsureSpatialRaycaster()
        {
            if (spatialRaycaster != null) return spatialRaycaster.isActiveAndEnabled;
            if (!EnvironmentRaycastManager.IsSupported) return false;
            spatialRaycaster = FindObjectOfType<EnvironmentRaycastManager>(true);
            if (spatialRaycaster != null) return spatialRaycaster.isActiveAndEnabled;
            ownedSpatialRaycaster = new GameObject("Quest spatial evidence raycaster");
            spatialRaycaster = ownedSpatialRaycaster.AddComponent<EnvironmentRaycastManager>();
            return true;
        }

        private IEnumerator WarmSpatialRaycaster()
        {
            // MRUK creates its native OpenXR raycaster asynchronously on enable.
            // Begin that work after VR focus and Scene permission, before the first A.
            while (spatialRaycaster == null)
            {
                if (OVRManager.isHmdPresent && OVRManager.hasVrFocus &&
                    OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
                {
                    var initialized = false;
                    var inactive = false;
                    try
                    {
                        initialized = EnsureSpatialRaycaster();
                        inactive = spatialRaycaster != null && !spatialRaycaster.isActiveAndEnabled;
                    }
                    catch (Exception error)
                    {
                        if (!spatialWarmupLogged)
                        {
                            Debug.LogWarning("QuestSpatialEvidence: warm-up deferred: " + error.GetType().Name);
                            spatialWarmupLogged = true;
                        }
                    }
                    if (initialized)
                    {
                        Debug.Log("QuestSpatialEvidence: raycaster enabled; waiting for Meta OpenXR handle");
                        yield break;
                    }
                    if (inactive)
                    {
                        Debug.LogWarning("QuestSpatialEvidence: existing raycaster is disabled; spatial evidence not ready");
                        yield break;
                    }
                    if (!spatialWarmupLogged)
                    {
                        Debug.Log("QuestSpatialEvidence: environment raycast unsupported or not initialized yet");
                        spatialWarmupLogged = true;
                    }
                }
                yield return new WaitForSecondsRealtime(2f);
            }
        }

        private byte[] CaptureSpatialEvidence(string captureId, DateTime rgbTimestamp, double triggerRealtime)
        {
            var started = Time.realtimeSinceStartupAsDouble;
            // The server rejects evidence outside this measured association window.
            // Preserve the RGB photo without a sidecar rather than inventing times.
            if (started - triggerRealtime > 0.5) return null;
            var evidence = new SpatialEvidence
            {
                capture_id = captureId,
                project_id = config.project_id,
                session_id = config.session_id,
                tracking_origin_id = config.tracking_origin_id,
                rgb_timestamp_raw = rgbTimestamp.ToString("O", CultureInfo.InvariantCulture),
                rgb_timestamp_kind = rgbTimestamp.Kind.ToString(),
                trigger_realtime_s = triggerRealtime,
                sample_started_realtime_s = started,
                sample_completed_realtime_s = started,
                availability = "not_ready"
            };
            try
            {
                if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
                    evidence.availability = "permission_denied";
                else if (!EnvironmentRaycastManager.IsSupported)
                    evidence.availability = "not_supported";
                else if (!AnchorReady || statusCamera == null || !EnsureSpatialRaycaster())
                    evidence.availability = "not_ready";
                else
                {
                    // This is an HMD angular survey near the RGB capture, not a calibrated
                    // per-pixel depth image. Every origin/direction is explicit in session space.
                    var generation = configGeneration;
                    var originMatrix = sessionOrigin.worldToLocalMatrix;
                    var headPosition = statusCamera.transform.position;
                    var headRotation = statusCamera.transform.rotation;
                    var fov = Mathf.Clamp(statusCamera.fieldOfView, 30f, 120f);
                    var tangentY = Mathf.Tan(fov * Mathf.Deg2Rad * 0.5f);
                    var tangentX = tangentY * Mathf.Clamp(statusCamera.aspect, 0.5f, 2.5f);
                    var targetCount = SpatialColumns * SpatialRows;
                    var rays = new SpatialRay[targetCount];
                    var count = 0;
                    var ready = false;
                    for (var sampleIndex = 0; sampleIndex < targetCount; sampleIndex++)
                    {
                        if (Time.realtimeSinceStartupAsDouble - started > SpatialTimeBudgetSeconds)
                            break;
                        var gridIndex = (sampleIndex * SpatialPermutationStride) % targetCount;
                        var row = gridIndex / SpatialColumns;
                        var col = gridIndex % SpatialColumns;
                        var x = ((col + 0.5f) / SpatialColumns * 2f - 1f) * tangentX;
                        var y = (1f - (row + 0.5f) / SpatialRows * 2f) * tangentY;
                        var worldDirection = (headRotation * new Vector3(x, y, 1f).normalized).normalized;
                        var rayOrigin = originMatrix.MultiplyPoint3x4(headPosition);
                        var rayDirection = originMatrix.MultiplyVector(worldDirection).normalized;
                        if (!Finite(rayOrigin.x) || !Finite(rayOrigin.y) || !Finite(rayOrigin.z) ||
                            rayOrigin.magnitude > 10000f || !Finite(rayDirection.x) ||
                            !Finite(rayDirection.y) || !Finite(rayDirection.z))
                            throw new InvalidDataException("Invalid spatial ray frame");
                        var sample = new SpatialRay
                        {
                            origin = Triple(rayOrigin),
                            direction = Triple(rayDirection),
                            status = "other"
                        };
                        var hitFound = spatialRaycaster.Raycast(new Ray(headPosition, worldDirection),
                            out var hit, 20f);
                        switch (hit.status)
                        {
                            case EnvironmentRaycastHitStatus.Hit:
                                if (hitFound && hit.normalConfidence > 0f &&
                                    hit.normal.sqrMagnitude > 0.25f &&
                                    Finite(hit.point.x) && Finite(hit.point.y) && Finite(hit.point.z))
                                {
                                    var sessionHit = originMatrix.MultiplyPoint3x4(hit.point);
                                    var along = Vector3.Dot(sessionHit - rayOrigin, rayDirection);
                                    var sideways = ((sessionHit - rayOrigin) - along * rayDirection).magnitude;
                                    if (Finite(along) && Finite(sideways) && along >= 0f && along <= 20f &&
                                        sideways <= Mathf.Max(0.05f, along * 0.01f))
                                    {
                                        sample.status = "hit";
                                        sample.hit = Triple(sessionHit);
                                        sample.normal = Triple(originMatrix.MultiplyVector(hit.normal).normalized);
                                        sample.normal_confidence = Mathf.Clamp01(hit.normalConfidence);
                                        ready = true;
                                    }
                                }
                                break;
                            case EnvironmentRaycastHitStatus.NoHit: sample.status = "no_hit"; ready = true; break;
                            case EnvironmentRaycastHitStatus.NotReady: sample.status = "not_ready"; break;
                            case EnvironmentRaycastHitStatus.HitPointOutsideOfCameraFrustum:
                                sample.status = "outside_frustum"; ready = true; break;
                            case EnvironmentRaycastHitStatus.HitPointOccluded:
                            case EnvironmentRaycastHitStatus.RayOccluded:
                                sample.status = "occluded"; ready = true; break;
                        }
                        rays[count++] = sample;
                    }
                    evidence.sample_completed_realtime_s = Time.realtimeSinceStartupAsDouble;
                    var moved = Vector3.Distance(headPosition, statusCamera.transform.position) > 0.03f ||
                        Quaternion.Angle(headRotation, statusCamera.transform.rotation) > 3f;
                    // RGB pixels were checked immediately after GetColors. This
                    // separately timed world survey is near-capture and unaligned;
                    // a later RGB frame naturally arrives during a dense sample.
                    var currentOrigin = sessionOrigin.worldToLocalMatrix;
                    var originChanged = (originMatrix.GetColumn(3) - currentOrigin.GetColumn(3)).sqrMagnitude > 0.000001f ||
                        (originMatrix.GetColumn(0) - currentOrigin.GetColumn(0)).sqrMagnitude > 0.000001f ||
                        (originMatrix.GetColumn(1) - currentOrigin.GetColumn(1)).sqrMagnitude > 0.000001f ||
                        (originMatrix.GetColumn(2) - currentOrigin.GetColumn(2)).sqrMagnitude > 0.000001f;
                    if (moved || originChanged || generation != configGeneration || !AnchorReady ||
                        evidence.sample_completed_realtime_s - started > 0.25)
                        evidence.availability = "sampling_error";
                    else if (ready)
                    {
                        evidence.availability = "ready";
                        if (count != rays.Length) Array.Resize(ref rays, count);
                        evidence.rays = rays;
                    }
                }
            }
            catch (Exception error)
            {
                Debug.LogWarning("QuestSpatialEvidence: " + error.GetType().Name);
                evidence.availability = "sampling_error";
                evidence.rays = Array.Empty<SpatialRay>();
            }
            evidence.sample_completed_realtime_s = Time.realtimeSinceStartupAsDouble;
            if (evidence.sample_completed_realtime_s - started > 0.25)
            {
                return null;
            }
            return SerializeSpatial(evidence);
        }
    }
}
