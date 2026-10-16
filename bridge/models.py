from __future__ import annotations

import math
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class NameBody(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 120:
            raise ValueError("Name must contain 1 to 120 characters")
        return value


class ProjectCreate(NameBody):
    request_id: uuid.UUID | None = None


class EntryCreate(NameBody):
    kind: Literal["category", "design", "variation"]
    parent_id: uuid.UUID | None = None
    sort_order: int = 0


class EntryPatch(BaseModel):
    name: str | None = None
    parent_id: uuid.UUID | None = None
    sort_order: int | None = None

    @field_validator("name")
    @classmethod
    def valid_optional_name(cls, value: str | None) -> str | None:
        return NameBody(name=value).name if value is not None else None

    @model_validator(mode="after")
    def no_null_updates(self):
        for field in ("name", "sort_order"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class SessionCreate(BaseModel):
    project_id: uuid.UUID
    tracking_origin_id: uuid.UUID
    request_id: uuid.UUID | None = None


class Intrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float

    @model_validator(mode="after")
    def valid(self):
        if not all(math.isfinite(x) for x in (self.fx, self.fy, self.cx, self.cy)) or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Intrinsics must be finite with positive focal lengths")
        return self


class CaptureMetadata(BaseModel):
    schema_version: Literal[1]
    capture_id: uuid.UUID
    project_id: uuid.UUID
    session_id: uuid.UUID
    tracking_origin_id: uuid.UUID
    camera_id: str = Field(min_length=1, max_length=64)
    image_timestamp_raw: str
    image_timestamp_kind: Literal["Utc", "Local", "Unspecified"]
    timestamp_clock: Literal["meta_datetime"]
    image_width: int = Field(gt=0, le=32768)
    image_height: int = Field(gt=0, le=32768)
    image_encoding: Literal["jpeg", "png"]
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spatial_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    intrinsics: Intrinsics
    camera_to_session: list[float] = Field(min_length=16, max_length=16)
    coordinate_convention: Literal["unity_lh_xright_yup_zforward"]
    position_units: Literal["meters"]
    tracking_valid: bool
    trigger_timestamp_ns: int | None = Field(default=None, ge=0)
    image_orientation: Literal["upright", "unity_texture_native"] = "upright"
    calibration_revision: str | None = Field(default=None, max_length=120)
    distortion: dict | None = None
    exposure_duration_ns: int | None = Field(default=None, ge=0)
    capture_source: Literal["quest_camera", "synthetic_test"] = "quest_camera"

    @model_validator(mode="after")
    def valid(self):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}(?:Z|[+-]\d{2}:\d{2})?", self.image_timestamp_raw):
            raise ValueError("image_timestamp_raw must use DateTime roundtrip ISO 8601 format")
        from datetime import datetime
        try:
            datetime.fromisoformat(self.image_timestamp_raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Invalid image timestamp") from error
        suffix = self.image_timestamp_raw[-1] == "Z"
        offset = bool(re.search(r"[+-]\d{2}:\d{2}$", self.image_timestamp_raw))
        if ((self.image_timestamp_kind == "Utc" and not suffix)
                or (self.image_timestamp_kind == "Local" and not offset)
                or (self.image_timestamp_kind == "Unspecified" and (suffix or offset))):
            raise ValueError("Timestamp suffix and DateTime kind disagree")
        if not self.tracking_valid:
            raise ValueError("Tracking must be valid")
        m = self.camera_to_session
        if not all(math.isfinite(x) for x in m):
            raise ValueError("Camera pose contains non-finite values")
        if any(abs(m[i] - expected) > 1e-4 for i, expected in ((12, 0), (13, 0), (14, 0), (15, 1))):
            raise ValueError("Camera pose has invalid homogeneous row")
        r = ((m[0], m[1], m[2]), (m[4], m[5], m[6]), (m[8], m[9], m[10]))
        for i in range(3):
            for j in range(3):
                dot = sum(r[i][k] * r[j][k] for k in range(3))
                if abs(dot - (1.0 if i == j else 0.0)) > 1e-3:
                    raise ValueError("Camera pose rotation is not orthonormal")
        determinant = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                       - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                       + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
        if abs(determinant - 1.0) > 1e-3:
            raise ValueError("Camera pose rotation must be proper")
        if self.distortion is not None:
            def finite_tree(value):
                if isinstance(value, dict):
                    return all(finite_tree(v) for v in value.values())
                if isinstance(value, list):
                    return all(finite_tree(v) for v in value)
                if isinstance(value, (float, int)):
                    return math.isfinite(value)
                return isinstance(value, (str, bool)) or value is None
            if not finite_tree(self.distortion):
                raise ValueError("Distortion contains non-finite numbers")
        return self


def _finite_vec3(values: list[float], *, bound: float = 10000.) -> bool:
    return len(values) == 3 and all(math.isfinite(value) and abs(value) <= bound for value in values)


class SpatialRay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: list[float] = Field(min_length=3, max_length=3)
    direction: list[float] = Field(min_length=3, max_length=3)
    status: Literal["hit", "no_hit", "not_ready", "outside_frustum", "occluded", "other"]
    hit: list[float] | None = Field(default=None, min_length=3, max_length=3)
    normal: list[float] | None = Field(default=None, min_length=3, max_length=3)
    normal_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def valid(self):
        if not _finite_vec3(self.origin) or not _finite_vec3(self.direction, bound=1.):
            raise ValueError("Spatial ray has invalid origin or direction")
        direction_length = math.sqrt(sum(value * value for value in self.direction))
        if abs(direction_length - 1.) > .01:
            raise ValueError("Spatial ray direction must be unit length")
        if self.status != "hit":
            if self.hit is not None or self.normal is not None or self.normal_confidence is not None:
                raise ValueError("Non-hit ray cannot include a hit or surface normal")
            return self
        if self.hit is None or not _finite_vec3(self.hit):
            raise ValueError("Hit ray requires finite hit position")
        if self.normal is not None:
            if not _finite_vec3(self.normal, bound=1.):
                raise ValueError("Spatial surface normal is invalid")
            normal_length = math.sqrt(sum(value * value for value in self.normal))
            if abs(normal_length - 1.) > .1:
                raise ValueError("Spatial surface normal must be unit length")
        if self.normal_confidence is not None and (
                not math.isfinite(self.normal_confidence) or self.normal is None):
            raise ValueError("Normal confidence requires a finite normal")
        displacement = [hit - origin for hit, origin in zip(self.hit, self.origin)]
        distance = sum(value * ray for value, ray in zip(displacement, self.direction))
        if distance < 0 or distance > 20.:
            raise ValueError("Spatial hit exceeds requested ray range")
        sideways = math.sqrt(sum((value - distance * ray) ** 2
                                 for value, ray in zip(displacement, self.direction)))
        if sideways > max(.05, .01 * distance):
            raise ValueError("Spatial hit is not on its requested ray")
        return self


class SpatialEvidence(BaseModel):
    """Immutable, bounded Quest environment-raycast observations per RGB capture."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    capture_id: uuid.UUID
    project_id: uuid.UUID
    session_id: uuid.UUID
    tracking_origin_id: uuid.UUID
    source: Literal["meta_environment_raycast_openxr", "synthetic_test"]
    association: Literal["near_capture_unaligned"]
    coordinate_frame: Literal["quest_session"]
    coordinate_convention: Literal["unity_lh_xright_yup_zforward"]
    position_units: Literal["meters"]
    trigger_realtime_s: float
    sample_started_realtime_s: float
    sample_completed_realtime_s: float
    rgb_timestamp_raw: str
    rgb_timestamp_kind: Literal["Utc", "Local", "Unspecified"]
    availability: Literal["ready", "not_supported", "permission_denied", "not_ready", "sampling_error"]
    head_to_session: list[float] | None = Field(default=None, min_length=16, max_length=16)
    rays: list[SpatialRay] = Field(max_length=4096)

    @model_validator(mode="after")
    def valid(self):
        if (not math.isfinite(self.trigger_realtime_s) or
                not math.isfinite(self.sample_started_realtime_s) or
                not math.isfinite(self.sample_completed_realtime_s) or
                self.trigger_realtime_s < 0 or
                self.sample_started_realtime_s < 0 or
                not 0 <= self.sample_started_realtime_s - self.trigger_realtime_s <= .5 or
                not 0 <= self.sample_completed_realtime_s - self.sample_started_realtime_s <= .25):
            raise ValueError("Spatial sampling interval is invalid")
        if self.availability != "ready" and self.rays:
            raise ValueError("Unavailable spatial sampling cannot report rays")
        if self.availability == "ready" and not self.rays:
            raise ValueError("Ready spatial sampling must report attempted rays")
        if self.head_to_session is not None:
            m = self.head_to_session
            if not all(math.isfinite(x) for x in m):
                raise ValueError("Head pose contains non-finite values")
            if any(abs(m[i] - expected) > 1e-4 for i, expected in
                   ((12, 0), (13, 0), (14, 0), (15, 1))):
                raise ValueError("Head pose has invalid homogeneous row")
            r = ((m[0], m[1], m[2]), (m[4], m[5], m[6]), (m[8], m[9], m[10]))
            for i in range(3):
                for j in range(3):
                    dot = sum(r[i][k] * r[j][k] for k in range(3))
                    if abs(dot - (1.0 if i == j else 0.0)) > 1e-3:
                        raise ValueError("Head pose rotation is not orthonormal")
            determinant = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
            if abs(determinant - 1.0) > 1e-3:
                raise ValueError("Head pose rotation must be proper")
        return self
