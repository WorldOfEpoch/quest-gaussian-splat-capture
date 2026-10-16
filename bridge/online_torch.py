"""Optional CUDA/gsplat engine for one continuously optimized Gaussian model.

Import only inside the offline worker environment with torch, gsplat and
plyfile installed.  The HTTP bridge does not import these GPU dependencies.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .online import OnlineError, triangulate_correspondences
from .region_edit import validate_region, region_membership, regions_overlap


SH_C0 = 0.28209479177387814


class GsplatTrainer:
    """One optimizer and Gaussian parameter set across all accepted views.

    Existing, real seed geometry is mandatory. Verified overlapping views
    can add bounded triangulated geometry to the same optimizer. Camera
    registration remains the coordinator's separate admission gate.
    """

    review_method_revision = "triangulated_patch_holdout_v1"

    def __init__(self, seed_ply: Path, checkpoint: Path | None = None,
                 *, published_ply: Path | None = None,
                 max_image_edge: int = 640, learning_rate: float = 0.001):
        try:
            import torch
            from gsplat.rendering import rasterization
            from plyfile import PlyData
        except ImportError as error:
            raise OnlineError("Online CUDA worker requires torch, gsplat and plyfile") from error
        if not torch.cuda.is_available():
            raise OnlineError("Online Gaussian optimization requires a CUDA GPU")
        self.torch = torch
        self.rasterization = rasterization
        self.device = torch.device("cuda")
        if not 128 <= max_image_edge <= 2048:
            raise OnlineError("Invalid image cache resolution")
        self.max_image_edge = max_image_edge
        self.max_cached_views = 24
        self.images: list[dict] = []
        self.accepted_ids: list[str] = []
        self.published_ids: list[str] = []
        self.candidate_capture_ids: list[str] = []
        self.rejected_capture_reviews: dict[str, dict] = {}
        self.added_total = 0
        self.last_growth_diagnostics: dict[str, int] = {}
        self.last_candidate_support: dict[str, list[int]] = {}
        self.candidate_observations: dict[str, list[int]] = {}
        self.candidate_validation_ids: list[str] = []
        self.candidate_training_ids: list[str] = []
        self.candidate_initial_scales = np.empty((0, 3), dtype=np.float32)
        self.review_attempts: dict[str, str] = {}
        self.pending_newest = False
        self.model_units_per_meter = 1.
        self.camera_distortion: np.ndarray | None = None
        self._camera_cache_requires_prime = False
        self.candidate_steps = 0
        self.minimum_candidate_steps = 64
        self.maximum_candidate_steps = 192
        self.last_publication_review = {"status": "seed_unchanged"}
        self.step = 0
        ply = PlyData.read(str(seed_ply))
        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or ())
        needed = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
                  "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
        if not needed <= names or len(vertex) < 3:
            raise OnlineError("Seed must be a nonempty Gaussian PLY")
        self._seed_vertex = vertex.copy()
        self._published_vertex = vertex.copy()
        self._published_ply_path = seed_ply
        def columns(keys):
            return np.stack([np.asarray(vertex[key], dtype=np.float32) for key in keys], axis=-1)
        means = columns(("x", "y", "z"))
        log_scales = columns(("scale_0", "scale_1", "scale_2"))
        quats = columns(("rot_0", "rot_1", "rot_2", "rot_3"))
        logits = columns(("opacity",))
        rgb = np.clip(0.5 + SH_C0 * columns(("f_dc_0", "f_dc_1", "f_dc_2")), 0.001, 0.999)
        if not all(np.isfinite(x).all() for x in (means, log_scales, quats, logits, rgb)):
            raise OnlineError("Seed Gaussian PLY contains non-finite parameters")
        self.seed_count = len(vertex)
        # Opt-in reviewed-region refinement. The default is still immutable
        # existing geometry; an older schema-1 checkpoint has no editable ROI.
        self.roi_editable_rows = torch.zeros(self.seed_count, dtype=torch.bool, device=self.device)
        self.roi_children_total = 0
        self.roi_gradient_sum = torch.zeros(self.seed_count, device=self.device)
        self.roi_gradient_count = torch.zeros(self.seed_count, device=self.device)
        self.removed_rows = torch.zeros(self.seed_count, dtype=torch.bool, device=self.device)
        self.locked_regions: list[dict] = []
        self.active_replacement: dict | None = None
        arrays = {
            "means": torch.nn.Parameter(torch.from_numpy(means).to(self.device)),
            "log_scales": torch.nn.Parameter(torch.from_numpy(log_scales).to(self.device)),
            "quats": torch.nn.Parameter(torch.from_numpy(quats).to(self.device)),
            "opacity_logits": torch.nn.Parameter(torch.from_numpy(logits[:, 0]).to(self.device)),
            "color_logits": torch.nn.Parameter(torch.from_numpy(np.log(rgb / (1 - rgb))).to(self.device)),
        }
        self.params = torch.nn.ParameterDict(arrays)
        self.candidate_anchors = self.params["means"].detach().clone()
        # For a fresh model, the input seed is the protected published anchor.
        # A resume replaces this anchor with the exact last published PLY;
        # reverting a trained checkpoint to the original seed changes the
        # visible model before any new image has been accepted.
        self._seed_parameters = {name: value.detach().clone()
                                 for name, value in self.params.items()}
        self.optimizer = torch.optim.Adam(self.params.parameters(), lr=learning_rate)
        if checkpoint and checkpoint.is_file():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            schema_version = payload.get("schema_version")
            if schema_version not in {1, 2, 3, 4} or payload.get("seed_count") != len(vertex):
                raise OnlineError("Online checkpoint is incompatible with seed geometry")
            restored = payload["parameters"]
            point_count = int(restored["means"].shape[0])
            if point_count < self.seed_count or point_count > 2_000_000 or any(
                    restored[key].shape != (point_count,) + value.shape[1:]
                    for key, value in self.params.items()) or any(
                        not torch.isfinite(value).all() for value in restored.values()):
                raise OnlineError("Online checkpoint has invalid Gaussian parameter dimensions")
            if point_count != self.seed_count:
                self.params = torch.nn.ParameterDict({
                    key: torch.nn.Parameter(restored[key].to(self.device))
                    for key in self.params.keys()})
                self.optimizer = torch.optim.Adam(self.params.parameters(), lr=learning_rate)
            self.params.load_state_dict(payload["parameters"])
            anchors = payload.get("candidate_anchors", self.params["means"].detach().cpu())
            if (not isinstance(anchors, torch.Tensor) or anchors.shape != (point_count, 3) or
                    not torch.isfinite(anchors).all()):
                raise OnlineError("Online checkpoint has invalid candidate triangulation anchors")
            self.candidate_anchors = anchors.to(self.device)
            self.optimizer.load_state_dict(payload["optimizer"])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
            if payload.get("camera_distortion") is not None:
                self.configure_camera_distortion(payload["camera_distortion"])
            self.images = payload["images"]
            if schema_version >= 4:
                self.configure_geometry_scale(payload.get("model_units_per_meter", 1.))
            self.accepted_ids = payload.get("accepted_ids", [item["capture_id"] for item in self.images])
            self.added_total = int(payload.get("added_total", sum(
                item.get("new_gaussians", 0) for item in self.images)))
            if len(set(self.accepted_ids)) != len(self.accepted_ids):
                raise OnlineError("Online checkpoint has duplicate capture receipts")
            self.published_ids = payload.get("published_capture_ids", self.accepted_ids if schema_version < 4 else [])
            self.candidate_capture_ids = payload.get("candidate_capture_ids", [])
            self.rejected_capture_reviews = payload.get("rejected_capture_reviews", {})
            self.candidate_observations = payload.get("candidate_observations", {})
            self.candidate_validation_ids = payload.get("candidate_validation_ids", [])
            self.candidate_training_ids = payload.get("candidate_training_ids", [])
            self.candidate_initial_scales = np.asarray(payload.get("candidate_initial_scales", []),
                                                      dtype=np.float32).reshape(-1, 3)
            self.review_attempts = payload.get("review_attempts", {})
            if (any(not isinstance(ids, list) or any(not isinstance(value, str) for value in ids) or
                    len(set(ids)) != len(ids) or not set(ids) <= set(self.accepted_ids)
                    for ids in (self.published_ids, self.candidate_capture_ids)) or
                    set(self.published_ids) & set(self.candidate_capture_ids) or
                    not isinstance(self.rejected_capture_reviews, dict) or
                    not set(self.rejected_capture_reviews) <= set(self.accepted_ids)):
                raise OnlineError("Online checkpoint has invalid publication capture lineage")
            self.images = self.images[-self.max_cached_views:]
            self.step = int(payload["step"])
            self.pending_newest = bool(payload.get("pending_newest", False))
            if schema_version >= 4:
                self.candidate_steps = int(payload.get("candidate_steps", 0))
                if not 0 <= self.candidate_steps <= self.maximum_candidate_steps:
                    raise OnlineError("Online checkpoint has invalid pending refinement state")
                self.last_publication_review = payload.get("publication_review", {})
            mask = payload.get("roi_editable_rows")
            if schema_version >= 2 and mask is None:
                raise OnlineError("Online checkpoint is missing ROI row lineage")
            if mask is not None:
                if mask.shape != (point_count,) or mask.dtype != torch.bool:
                    raise OnlineError("Online checkpoint has invalid ROI row lineage")
                self.roi_editable_rows = mask.to(self.device)
                self.roi_children_total = int(payload.get("roi_children_total", 0))
                self.roi_gradient_sum = payload.get(
                    "roi_gradient_sum", torch.zeros(point_count)).to(self.device)
                self.roi_gradient_count = payload.get(
                    "roi_gradient_count", torch.zeros(point_count)).to(self.device)
                if self.roi_gradient_sum.shape != (point_count,) or self.roi_gradient_count.shape != (point_count,):
                    raise OnlineError("Online checkpoint has invalid ROI gradient state")
                if self.roi_editable_rows.any():
                    if payload.get("roi_anchor_guard_version") != 1 or "candidate_anchors" not in payload:
                        raise OnlineError("Reviewed ROI checkpoint lacks fixed refinement anchors")
                    self._validate_reviewed_anchor_bound()
            else:
                self.roi_editable_rows = torch.zeros(point_count, dtype=torch.bool, device=self.device)
                self.roi_gradient_sum = torch.zeros(point_count, device=self.device)
                self.roi_gradient_count = torch.zeros(point_count, device=self.device)
            self.removed_rows = torch.zeros(point_count, dtype=torch.bool, device=self.device)
            if schema_version >= 3:
                removed = payload.get("removed_rows")
                if not isinstance(removed, torch.Tensor) or removed.dtype != torch.bool or removed.shape != (point_count,):
                    raise OnlineError("Online checkpoint has invalid removed-row state")
                self.removed_rows = removed.to(self.device)
                self.locked_regions = [validate_region(r) for r in payload.get("locked_regions", [])]
                if len({r["id"] for r in self.locked_regions}) != len(self.locked_regions):
                    raise OnlineError("Online checkpoint has duplicate area locks")
                active = payload.get("active_replacement")
                if active is not None:
                    region = validate_region(active["region"])
                    allowed, used = active.get("capture_ids"), active.get("used_ids")
                    if (not isinstance(allowed, list) or not isinstance(used, list) or
                            not all(isinstance(x, str) for x in allowed + used) or
                            len(set(allowed)) != len(allowed) or len(set(used)) != len(used) or
                            not set(used) <= set(allowed) or
                            any(regions_overlap(region, r) for r in self.locked_regions)):
                        raise OnlineError("Online checkpoint has invalid area replacement state")
                    self.active_replacement = {"region": region, "capture_ids": allowed, "used_ids": used}
            published = published_ply or checkpoint.with_name("current.ply")
            if not published.is_file():
                raise OnlineError("A resumed checkpoint needs its published Gaussian PLY")
            published_vertex = PlyData.read(str(published))["vertex"].data
            published_count = len(published_vertex)
            if ((schema_version < 4 and published_count != point_count) or
                    not self.seed_count <= published_count <= point_count or
                    (schema_version >= 4 and payload.get("published_count") != published_count) or
                    not needed <= set(published_vertex.dtype.names or ())):
                raise OnlineError("Published Gaussian PLY does not match checkpoint dimensions")
            # A crash while replacing one side of the pair must fail closed.
            # The old renderer wrote normalized quaternions and converted
            # bounded RGB logits to DC coefficients, so compare those forms.
            stored = {name: value[:published_count].detach().cpu().numpy() for name, value in self.params.items()}
            checks = (
                (stored["means"], np.stack([published_vertex[n] for n in ("x", "y", "z")], axis=-1)),
                (stored["log_scales"], np.stack([published_vertex[f"scale_{i}"] for i in range(3)], axis=-1)),
                (stored["opacity_logits"], np.asarray(published_vertex["opacity"])),
            )
            if any(not np.allclose(left, right, rtol=1e-5, atol=1e-5) for left, right in checks):
                raise OnlineError("Published Gaussian PLY and checkpoint disagree")
            stored_quats = stored["quats"] / np.maximum(np.linalg.norm(stored["quats"], axis=1, keepdims=True), 1e-8)
            ply_quats = np.stack([published_vertex[f"rot_{i}"] for i in range(4)], axis=-1)
            ply_quats /= np.maximum(np.linalg.norm(ply_quats, axis=1, keepdims=True), 1e-8)
            if not np.allclose(stored_quats, ply_quats, rtol=1e-5, atol=1e-5):
                raise OnlineError("Published Gaussian rotations and checkpoint disagree")
            rgb = 1. / (1. + np.exp(-np.clip(stored["color_logits"], -30., 30.)))
            actual_dc = (rgb - .5) / SH_C0
            published_dc = np.stack([published_vertex[f"f_dc_{i}"] for i in range(3)], axis=-1)
            valid_dc = ((.001 <= .5 + SH_C0 * published_dc) &
                        (.5 + SH_C0 * published_dc <= .999)).all(axis=1)
            if not np.allclose(actual_dc[valid_dc], published_dc[valid_dc], rtol=1e-5, atol=1e-5):
                raise OnlineError("Published Gaussian colors and checkpoint disagree")
            self._published_vertex = published_vertex.copy()
            self._published_ply_path = published
            self._seed_vertex = published_vertex[:self.seed_count].copy()
            self._seed_parameters = {name: value[:self.seed_count].detach().clone()
                                     for name, value in self.params.items()}
        self._protect_seed()
        # Unpublished rows in schema-4 checkpoints remain candidates. They must
        # finish refinement and validation; a restart must not freeze them.
        self._protect_existing(len(self._published_vertex))
        self._cache_published_appearance()
        if self.candidate_observations:
            count = len(self.params["means"]) - self.protected_count
            cached_ids = {v["capture_id"] for v in self.images}
            if (self.candidate_initial_scales.shape != (count, 3) or
                    not np.isfinite(self.candidate_initial_scales).all() or
                    np.any(self.candidate_initial_scales <= 0) or
                    not set(self.candidate_validation_ids + self.candidate_training_ids) <= cached_ids or
                    set(self.candidate_validation_ids) & set(self.candidate_training_ids) or
                    any(not isinstance(indices, list) or not indices or
                        any(type(i) is not int or not 0 <= i < count for i in indices)
                        for indices in self.candidate_observations.values())):
                raise OnlineError("Checkpoint has invalid fixed candidate support")

    def _cache_published_appearance(self):
        """Retain complete seed SH appearance in the optimizer's quality renders."""
        fields = set(self._published_vertex.dtype.names)
        rest = [name for name in fields if name.startswith("f_rest_")]
        coefficient_count = 1 + len(rest) // 3
        degree = int(round(np.sqrt(coefficient_count))) - 1
        self.sh_degree = degree if (len(rest) % 3 == 0 and
            coefficient_count == (degree + 1) ** 2 and 0 <= degree <= 3 and
            all(f"f_rest_{i}" in fields for i in range(len(rest)))) else None
        if self.sh_degree is None:
            self._published_sh = None
            return
        dc = np.stack([self._published_vertex[f"f_dc_{i}"] for i in range(3)], axis=-1)[:, None, :]
        if rest:
            higher = np.stack([self._published_vertex[f"f_rest_{i}"] for i in range(len(rest))], axis=-1)
            higher = higher.reshape(len(dc), 3, coefficient_count - 1).transpose(0, 2, 1)
            dc = np.concatenate((dc, higher), axis=1)
        self._published_sh = self.torch.from_numpy(dc.astype(np.float32)).to(self.device)

    def _appearance(self, count):
        torch = self.torch
        rgb = torch.sigmoid(self.params["color_logits"][:count])
        if self.sh_degree is None:
            return rgb, {}
        old_count = min(count, len(self._published_sh))
        coefficients = torch.cat((self._published_sh[:old_count], torch.zeros(
            (count - old_count, (self.sh_degree + 1) ** 2, 3), device=self.device)), dim=0).clone()
        coefficients[:, 0, :] = (rgb - .5) / SH_C0
        frozen = self._frozen_rows[:old_count]
        coefficients[:old_count, 0, :][frozen] = self._published_sh[:old_count, 0, :][frozen]
        return coefficients, {"sh_degree": self.sh_degree}

    def configure_geometry_scale(self, model_units_per_meter: float) -> None:
        """Use the verified session-to-model similarity scale for physical limits."""
        try:
            value = float(model_units_per_meter)
        except (TypeError, ValueError) as error:
            raise OnlineError("Invalid verified model geometry scale") from error
        if not np.isfinite(value) or not .01 <= value <= 100.:
            raise OnlineError("Invalid verified model geometry scale")
        if self.accepted_ids and not np.isclose(value, self.model_units_per_meter, rtol=1e-7):
            raise OnlineError("Cannot change geometry scale after photo admission")
        self.model_units_per_meter = value

    def configure_camera_distortion(self, coefficients) -> None:
        """Use one measured OpenCV camera calibration for the working cache.

        A legacy pinhole cache may adopt calibration only before admitting a
        photo or optimizing. Its baseline cameras must then be re-primed.
        """
        try:
            value = np.asarray(coefficients, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise OnlineError("Invalid measured camera distortion") from error
        if value.shape != (5,) or not np.isfinite(value).all() or np.max(np.abs(value)) > 2.:
            raise OnlineError("Invalid measured camera distortion")
        if self.camera_distortion is not None:
            if not np.array_equal(value, self.camera_distortion):
                raise OnlineError("Cannot change the established camera distortion")
            return
        if self.accepted_ids or self.step:
            raise OnlineError("Legacy camera calibration must precede photo admission and optimization")
        self.camera_distortion = value.copy()
        self._camera_cache_requires_prime = bool(self.images)

    def prime_baseline_views(self, views: list[dict]) -> int:
        """Cache verified seed cameras without changing geometry or photo receipts.

        The coordinator supplies registered OpenCV model-space poses. Two
        spatially separated baseline cameras are evaluation-only: they may
        support matching but never provide optimizer loss targets.
        """
        ids = [view.get("capture_id") for view in views]
        if len(views) < 3 or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise OnlineError("Baseline cache requires three distinct verified cameras")
        old_guard_ids = [view["capture_id"] for view in self.images if view.get("validation_only")]
        if old_guard_ids:
            if len(old_guard_ids) != 2 or len(set(old_guard_ids)) != 2 or any(i not in ids for i in old_guard_ids):
                raise OnlineError("Established held-out baseline cameras are missing or invalid")
            # Corrected measured poses can change the farthest pair, and uniform
            # resampling can omit an established guard. Keep their identities
            # and order while refreshing their verified camera data.
            guard_indices = [ids.index(i) for i in old_guard_ids]
            for cid in self.candidate_validation_ids:
                if cid in ids and ids.index(cid) not in guard_indices:
                    guard_indices.append(ids.index(cid))
                elif cid not in ids and any(v["capture_id"] == cid and v.get("baseline") for v in self.images):
                    raise OnlineError("Fixed candidate validation camera is missing")
            remaining = [i for i in range(len(views)) if i not in guard_indices]
            sampled = (np.unique(np.linspace(0, len(remaining) - 1,
                min(len(remaining), self.max_cached_views - len(guard_indices))).astype(int)) if remaining else [])
            selected = guard_indices + [remaining[i] for i in sampled]
        else:
            selected = np.unique(np.linspace(0, len(views) - 1,
                min(len(views), self.max_cached_views)).astype(int))
        cached = []
        for index in selected:
            view = views[index]
            pose, k = self._validated_view(view["pose"], view["intrinsics"])
            pixels, scaled_k = self._scaled_image(Path(view["image_path"]), k)
            cached.append({"capture_id": view["capture_id"], "new_gaussians": 0,
                           "baseline": True, "validation_only": False,
                           "pixels": self.torch.from_numpy(pixels),
                           "pose": self.torch.from_numpy(pose.astype(np.float32)),
                           "intrinsics": self.torch.from_numpy(scaled_k.astype(np.float32))})
        centers = np.array([view["pose"].numpy()[:3, 3] for view in cached])
        if old_guard_ids:
            left, right = 0, 1
        else:
            distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
            left, right = np.unravel_index(distance.argmax(), distance.shape)
        if np.linalg.norm(centers[left] - centers[right]) < .08 * self.model_units_per_meter:
            raise OnlineError("Baseline validation cameras lack independent translated views")
        cached[left]["validation_only"] = cached[right]["validation_only"] = True
        retained = [view for view in self.images if not view.get("baseline") and
                    view["capture_id"] not in ids]
        guarded = [view for view in cached + retained if self._validation_view(view)]
        training = [view for view in cached + retained if not self._validation_view(view)]
        self.images = guarded + training[-(self.max_cached_views - len(guarded)):]
        self._camera_cache_requires_prime = False
        return len(cached)

    @staticmethod
    def _validated_view(pose, intrinsics):
        pose, k = np.asarray(pose, dtype=np.float64), np.asarray(intrinsics, dtype=np.float64)
        if (pose.shape != (4, 4) or k.shape != (3, 3) or
                not np.isfinite(pose).all() or not np.isfinite(k).all() or
                not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4) or
                min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-5)):
            raise OnlineError("Invalid verified online camera")
        return pose, k

    @property
    def has_pending_work(self) -> bool:
        return len(self.params["means"]) > self.protected_count

    @property
    def pending_refinement(self) -> dict:
        return {"pending": self.has_pending_work,
                "candidate_gaussians": len(self.params["means"]) - self.protected_count,
                "training_steps": self.candidate_steps,
                "minimum_training_steps": self.minimum_candidate_steps,
                "maximum_training_steps": self.maximum_candidate_steps,
                "capture_ids": list(self.candidate_capture_ids),
                "publication_review": self.last_publication_review}

    def _protect_seed(self) -> None:
        """Restore seed rows and erase Adam momentum that could move them."""
        torch = self.torch
        with torch.no_grad():
            for name, parameter in self.params.items():
                frozen = ~self.roi_editable_rows[:self.seed_count]
                parameter[:self.seed_count][frozen] = self._seed_parameters[name][frozen]
                state = self.optimizer.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    moment = state.get(key)
                    if isinstance(moment, torch.Tensor) and moment.ndim and moment.shape[0] >= self.seed_count:
                        moment[:self.seed_count][frozen] = 0

    def _protect_existing(self, count: int | None = None) -> None:
        """Freeze all published rows, including additions from earlier photos."""
        self._apply_region_locks()
        self.protected_count = len(self.params["means"]) if count is None else count
        self._frozen_rows = ~self.roi_editable_rows[:self.protected_count].clone()
        self._protected_parameters = {name: value[:self.protected_count].detach().clone()
                                      for name, value in self.params.items()}
        self._restore_protected()

    def _restore_protected(self) -> None:
        torch = self.torch
        with torch.no_grad():
            for name, parameter in self.params.items():
                parameter[:self.protected_count][self._frozen_rows] = (
                    self._protected_parameters[name][self._frozen_rows])
                state = self.optimizer.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    moment = state.get(key)
                    if isinstance(moment, torch.Tensor) and moment.ndim and moment.shape[0] >= self.protected_count:
                        moment[:self.protected_count][self._frozen_rows] = 0

    def _validate_reviewed_anchor_bound(self) -> None:
        """Never resume or save edited surfaces with an unproved moving origin."""
        ids = self.torch.where(self.roi_editable_rows)[0]
        if not len(ids):
            return
        delta = self.params["means"][ids].detach() - self.candidate_anchors[ids]
        length = self.torch.linalg.vector_norm(delta, dim=-1)
        limit = .03 * self.model_units_per_meter
        if not self.torch.isfinite(length).all() or (length > limit + 1e-6*max(1., limit)).any():
            raise OnlineError("Reviewed ROI geometry escaped its fixed physical refinement bound")

    def _bound_reviewed_positions(self) -> None:
        """Constrain reviewed seed/child positions to three centimeters total.

        Anchors are fixed when the region is reviewed, not after each save or
        restart. Region containment and locks may impose stricter bounds.
        """
        torch = self.torch
        ids = torch.where(self.roi_editable_rows & ~self.removed_rows)[0]
        if not len(ids):
            return
        with torch.no_grad():
            delta = self.params["means"][ids] - self.candidate_anchors[ids]
            length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            if not torch.isfinite(length).all():
                raise OnlineError("Reviewed ROI refinement produced non-finite geometry")
            limit = .03 * self.model_units_per_meter
            clamped = length[:, 0] > limit
            if not clamped.any():
                return
            changed = ids[clamped]
            self.params["means"][changed] = (self.candidate_anchors[changed] +
                delta[clamped] * (limit/length[clamped]))
            # Do not retain the outward Adam force after reaching the bound.
            state = self.optimizer.state.get(self.params["means"], {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moment = state.get(key)
                if isinstance(moment, torch.Tensor) and moment.ndim and len(moment) == len(self.params["means"]):
                    moment[changed] = 0

    def _grow(self, points: np.ndarray, colors: np.ndarray, scales: np.ndarray) -> int:
        """Append Gaussians while retaining Adam moments for every old point."""
        if not len(points):
            return 0
        # Test the footprint that will actually be stored. The minimum render
        # scale can enlarge a tiny candidate enough to cross a locked boundary.
        units = getattr(self, "model_units_per_meter", 1.)
        scales = np.clip(scales, .003 * units, .08 * units)
        allowed = self._allowed_growth(points, scales)
        self._last_growth_selected_indices = np.flatnonzero(allowed)
        points, colors, scales = points[allowed], colors[allowed], scales[allowed]
        if not len(points):
            return 0
        torch = self.torch
        n = len(points)
        ones = np.zeros((n, 4), dtype=np.float32)
        ones[:, 0] = 1.
        colors = np.clip(colors, 0.001, 0.999)
        appended = {
            "means": points.astype(np.float32),
            "log_scales": np.log(scales).astype(np.float32),
            "quats": ones,
            "opacity_logits": np.full(n, -1.5, dtype=np.float32),
            "color_logits": np.log(colors / (1. - colors)).astype(np.float32),
        }
        previous = self.params
        old_optimizer = self.optimizer
        new_params = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.cat((parameter.detach(),
                                               torch.from_numpy(appended[name]).to(self.device)), dim=0))
            for name, parameter in previous.items()})
        optimizer = torch.optim.Adam(new_params.parameters(), lr=old_optimizer.param_groups[0]["lr"])
        for name, old_parameter in previous.items():
            state = old_optimizer.state.get(old_parameter)
            if not state:
                continue
            new_parameter = new_params[name]
            optimizer.state[new_parameter] = {
                key: torch.cat((value, torch.zeros_like(new_parameter.detach()[-n:])), dim=0)
                if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} else
                value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in state.items()}
        self.params = new_params
        self.optimizer = optimizer
        self.candidate_anchors = torch.cat((self.candidate_anchors,
            torch.from_numpy(points.astype(np.float32)).to(self.device)), dim=0)
        self.roi_editable_rows = torch.cat((self.roi_editable_rows,
            torch.full((n,), self.active_replacement is not None, dtype=torch.bool, device=self.device)))
        self.removed_rows = torch.cat((self.removed_rows,
                                       torch.zeros(n, dtype=torch.bool, device=self.device)))
        self.roi_gradient_sum = torch.cat((self.roi_gradient_sum, torch.zeros(n, device=self.device)))
        self.roi_gradient_count = torch.cat((self.roi_gradient_count, torch.zeros(n, device=self.device)))
        self.candidate_steps = 0
        return n

    def _geometry(self):
        return (self.params["means"].detach().cpu().numpy(),
                self.torch.exp(self.params["log_scales"].detach()).cpu().numpy())

    def _allowed_growth(self, points, scales):
        allowed = np.ones(len(points), dtype=bool)
        for region in self.locked_regions:
            allowed &= ~region_membership(points, scales, region)
        if self.active_replacement:
            allowed &= region_membership(points, scales, self.active_replacement["region"], contained=True)
        return allowed

    def _apply_region_locks(self):
        self.roi_editable_rows[self.removed_rows] = False
        if self.locked_regions:
            points, scales = self._geometry()
            for region in self.locked_regions:
                selected = region_membership(points, scales, region)
                self.roi_editable_rows[self.torch.from_numpy(selected).to(self.device)] = False

    def begin_area_replacement(self, region: dict, capture_ids: list[str]) -> dict:
        """Clear an isolated draft region; preserve row identity and all outside data.

        Removed rows become permanently invisible tombstones in this draft.
        Keeping their slots avoids renumbering checkpoint/seed records; the
        original published files are the independent undo source.
        """
        region = validate_region(region)
        if self.active_replacement or self.roi_editable_rows.any():
            raise OnlineError("Finish the current region edit before starting another")
        if (not isinstance(capture_ids, list) or len(capture_ids) > 128 or
                not all(isinstance(x, str) and x for x in capture_ids) or
                len(set(capture_ids)) != len(capture_ids)):
            raise OnlineError("Select at most 128 distinct replacement photos")
        if any(regions_overlap(region, old) for old in self.locked_regions):
            raise OnlineError("Area overlaps a locked region; explicitly unlock it first")
        points, scales = self._geometry()
        inside = region_membership(points, scales, region, contained=True)
        crossing = region_membership(points, scales, region) & ~inside
        indices = self.torch.from_numpy(inside).to(self.device)
        self.removed_rows |= indices
        with self.torch.no_grad():
            self.params["opacity_logits"][indices] = -30.
            self._seed_parameters["opacity_logits"][indices[:self.seed_count]] = -30.
        self._published_vertex["opacity"][inside[:len(self._published_vertex)]] = -30.
        self.roi_editable_rows.zero_()
        self.roi_gradient_sum.zero_()
        self.roi_gradient_count.zero_()
        # Old object images must not reseed the cleared space. Replacement
        # photos build their own bounded overlap cache in the same frame.
        self.images = []
        self.pending_newest = False
        self.active_replacement = {"region": region, "capture_ids": list(capture_ids), "used_ids": []}
        self._protect_existing()
        return {"removed_gaussians": int(inside.sum()), "preserved_boundary_gaussians": int(crossing.sum())}

    def protect_existing_area(self, region: dict) -> dict:
        """Lock accepted geometry in place without clearing or training it."""
        region = validate_region(region)
        if self.active_replacement or self.roi_editable_rows.any() or self.has_pending_work:
            raise OnlineError("Finish the current refinement before protecting an existing area")
        if any(regions_overlap(region, old) for old in self.locked_regions):
            raise OnlineError("Area overlaps an existing lock; explicitly unlock it first")
        self.locked_regions.append(region)
        self._protect_existing()
        return region

    def lock_replacement_area(self) -> dict:
        if not self.active_replacement:
            raise OnlineError("No replacement area is active")
        # Commit the latest editable values before marking them immutable.
        # Otherwise save() could copy an older published record over the
        # final training steps when locking between two saves.
        count = len(self._published_vertex)
        editable = self.roi_editable_rows[:count].cpu().numpy()
        with self.torch.no_grad():
            p = {name: value[:count].detach().float().cpu().numpy() for name, value in self.params.items()}
            quats = p["quats"] / np.maximum(np.linalg.norm(p["quats"], axis=1, keepdims=True), 1e-8)
            rgb = 1 / (1 + np.exp(-np.clip(p["color_logits"], -30, 30)))
            for names, values in ((["x", "y", "z"], p["means"]),
                                  (["f_dc_0", "f_dc_1", "f_dc_2"], (rgb - .5) / SH_C0),
                                  (["scale_0", "scale_1", "scale_2"], p["log_scales"]),
                                  (["rot_0", "rot_1", "rot_2", "rot_3"], quats)):
                for axis, name in enumerate(names):
                    self._published_vertex[name][editable] = values[editable, axis]
            self._published_vertex["opacity"][editable] = p["opacity_logits"][editable]
        region = self.active_replacement["region"]
        self.locked_regions.append(region)
        self.active_replacement = None
        self.roi_editable_rows.zero_()
        self._protect_existing()
        return region

    def unlock_area(self, region_id: str) -> None:
        if not any(r["id"] == region_id for r in self.locked_regions):
            raise OnlineError("Area lock was not found")
        self.locked_regions = [r for r in self.locked_regions if r["id"] != region_id]
        # Unlock permits an explicit new replacement, not global erosion of
        # previously published geometry. Removed rows stay removed.
        self._protect_existing()

    def configure_reviewed_roi(self, views: list[dict], *, max_seed_rows: int = 2048) -> int:
        """Opt in seed rows whose whole projected footprint is inside two edit views.

        Every view supplies model-space camera_to_world `pose`, pixel-space
        `intrinsics`, (width, height) `size`, and either `role='edit'` with a
        reviewed pixel rectangle or `role='protect'` for the whole image. A
        seed visible in a protect view remains immutable. No automatic call
        to this method is made by normal capture processing.
        """
        if self.active_replacement:
            raise OnlineError("A replacement area is already active")
        if self.roi_editable_rows.any():
            raise OnlineError("A reviewed ROI is already active")
        if not 1 <= max_seed_rows <= 4096:
            raise OnlineError("Invalid reviewed ROI seed cap")
        if sum(v.get("role") == "edit" for v in views) < 2 or not any(
                v.get("role") == "protect" for v in views):
            raise OnlineError("Reviewed ROI needs two edit views and one protect view")
        means = self.params["means"][:self.seed_count].detach().cpu().numpy().astype(np.float64)
        scale = np.exp(self.params["log_scales"][:self.seed_count].detach().cpu().numpy()).max(axis=1)
        inside_count = np.zeros(self.seed_count, dtype=np.int32)
        protected = np.zeros(self.seed_count, dtype=bool)
        radius_score = np.zeros(self.seed_count)
        for view in views:
            pose = np.asarray(view["pose"], dtype=np.float64)
            k = np.asarray(view["intrinsics"], dtype=np.float64)
            width, height = view["size"]
            role = view["role"]
            if pose.shape != (4, 4) or k.shape != (3, 3) or not all(
                    np.isfinite(x).all() for x in (pose, k)) or width < 1 or height < 1 or role not in {"edit", "protect"}:
                raise OnlineError("Invalid reviewed ROI camera")
            camera = (np.linalg.inv(pose) @ np.column_stack((means, np.ones(len(means)))).T).T[:, :3]
            depth = camera[:, 2]
            safe_depth = np.maximum(depth, 1e-6)
            projected = (k @ camera.T).T
            uv = projected[:, :2] / safe_depth[:, None]
            radius = 3. * scale * max(k[0, 0], k[1, 1]) / safe_depth
            visible = ((depth > .25 * self.model_units_per_meter) &
                       (depth < 20. * self.model_units_per_meter) & np.isfinite(radius) &
                       (uv[:, 0] + radius >= 0) & (uv[:, 0] - radius < width) &
                       (uv[:, 1] + radius >= 0) & (uv[:, 1] - radius < height))
            if role == "protect":
                protected |= visible
                continue
            rect = np.asarray(view.get("rect"), dtype=np.float64)
            if rect.shape != (4,) or not np.isfinite(rect).all() or not (
                    0 <= rect[0] < rect[2] <= width and 0 <= rect[1] < rect[3] <= height):
                raise OnlineError("Invalid reviewed ROI rectangle")
            contained = (visible & (uv[:, 0] - radius >= rect[0]) &
                         (uv[:, 0] + radius <= rect[2]) &
                         (uv[:, 1] - radius >= rect[1]) &
                         (uv[:, 1] + radius <= rect[3]))
            inside_count += contained
            radius_score = np.maximum(radius_score, np.where(contained, radius, 0.))
        eligible = np.flatnonzero((inside_count >= 2) & ~protected & (radius_score >= 1.))
        if len(eligible):
            points, scales = self._geometry()
            eligible = eligible[self._allowed_growth(points[eligible], scales[eligible]) &
                                ~self.removed_rows[eligible].cpu().numpy()]
        if len(eligible) > max_seed_rows:
            eligible = eligible[np.argsort(-radius_score[eligible], kind="stable")[:max_seed_rows]]
        if not len(eligible):
            raise OnlineError("Reviewed ROI contains no isolated seed Gaussians")
        with self.torch.no_grad():
            self.candidate_anchors[eligible] = self.params["means"][eligible].detach()
        self.roi_editable_rows[eligible] = True
        self._protect_existing()
        return len(eligible)

    def split_reviewed_roi(self, *, max_children: int = 64,
                           min_observations: int = 4) -> int:
        """Split only gradient-supported editable rows, preserving parent IDs.

        This is explicit and off by default. A child inherits the parent's
        appearance and is confined to the reviewed ROI lineage. Normal
        capture processing never invokes this method.
        """
        torch = self.torch
        if self.locked_regions or self.active_replacement:
            raise OnlineError("Region edits use bounded triangulation; unrestricted ROI splitting is disabled")
        remaining = min(512 - self.roi_children_total, 2_000_000 - len(self.params["means"]))
        if not 1 <= max_children <= 128 or not 1 <= min_observations <= 100 or remaining <= 0:
            return 0
        score = self.roi_gradient_sum / self.roi_gradient_count.clamp(min=1)
        eligible = (self.roi_editable_rows & (self.roi_gradient_count >= min_observations) &
                    torch.isfinite(score) & (score > .0002))
        indices = torch.where(eligible)[0]
        if not len(indices):
            return 0
        indices = indices[torch.argsort(score[indices], descending=True)[:min(max_children, remaining)]]
        n = len(indices)
        from gsplat.utils import normalized_quat_to_rotmat
        with torch.no_grad():
            old = {name: param.detach().clone() for name, param in self.params.items()}
            scales = torch.exp(old["log_scales"][indices])
            dominant = scales.argmax(dim=1)
            local_axis = torch.nn.functional.one_hot(dominant, num_classes=3).to(scales.dtype)
            rotation = normalized_quat_to_rotmat(
                torch.nn.functional.normalize(old["quats"][indices], dim=-1))
            direction = torch.bmm(rotation, local_axis[..., None]).squeeze(-1)
            offset = direction * scales.gather(1, dominant[:, None]) * .35
            children = {name: value[indices].clone() for name, value in old.items()}
            old["means"][indices] -= offset
            children["means"] += offset
            old["log_scales"][indices, dominant] -= np.log(1.6)
            children["log_scales"][torch.arange(n, device=self.device), dominant] -= np.log(1.6)
            alpha = torch.sigmoid(old["opacity_logits"][indices]).clamp(1e-5, 1 - 1e-5)
            half_alpha = (1 - torch.sqrt(1 - alpha)).clamp(1e-5, 1 - 1e-5)
            half_logit = torch.logit(half_alpha)
            old["opacity_logits"][indices] = half_logit
            children["opacity_logits"] = half_logit
            previous, previous_optimizer = self.params, self.optimizer
            new_params = torch.nn.ParameterDict({
                name: torch.nn.Parameter(torch.cat((value, children[name]), dim=0))
                for name, value in old.items()})
            optimizer = torch.optim.Adam(new_params.parameters(),
                                         lr=previous_optimizer.param_groups[0]["lr"])
            for name, previous_parameter in previous.items():
                state = previous_optimizer.state.get(previous_parameter)
                if not state:
                    continue
                copied = {}
                for key, value in state.items():
                    if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                        moment = value.clone()
                        moment[indices] = 0
                        copied[key] = torch.cat((moment, torch.zeros_like(moment[:n])), dim=0)
                    else:
                        copied[key] = value.clone() if isinstance(value, torch.Tensor) else value
                optimizer.state[new_params[name]] = copied
            self.params, self.optimizer = new_params, optimizer
            # A split inherits the original reviewed anchor. Repeated splits
            # cannot move the origin and accumulate another three centimeters.
            self.candidate_anchors = torch.cat((self.candidate_anchors,
                                               self.candidate_anchors[indices].clone()), dim=0)
            self.roi_editable_rows = torch.cat((self.roi_editable_rows,
                                                torch.ones(n, dtype=torch.bool, device=self.device)))
            self.removed_rows = torch.cat((self.removed_rows,
                                           torch.zeros(n, dtype=torch.bool, device=self.device)))
            self.roi_gradient_sum = torch.cat((self.roi_gradient_sum,
                                               torch.zeros(n, device=self.device)))
            self.roi_gradient_count = torch.cat((self.roi_gradient_count,
                                                 torch.zeros(n, device=self.device)))
            self.roi_gradient_sum[indices] = 0
            self.roi_gradient_count[indices] = 0
            self.roi_children_total += n
            self._bound_reviewed_positions()
        return n

    def _new_points(self, pixels: np.ndarray, pose: np.ndarray,
                    intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match a new view against overlapping cached views and find unseen space.

        Headset motion is not a straight path: a useful baseline can be more
        than four captures old. Compare the bounded 24-view cache, then spend
        triangulation work on the strongest descriptor-overlap pairs while
        always retaining eligible recent pairs. The geometric depth, parallax
        and reprojection checks remain in triangulate_correspondences.
        """
        diagnostics = {"cached_views": len(self.images), "baseline_views": 0,
                       "matched_pairs": 0, "triangulated_pairs": 0,
                       "triangulated_points": 0, "unoccupied_points": 0,
                       "added_points": 0}
        self.last_growth_diagnostics = diagnostics
        units = getattr(self, "model_units_per_meter", 1.)
        if not self.images or len(self.params["means"]) >= 2_000_000:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
        try:
            import cv2
            from scipy.spatial import cKDTree
        except ImportError as error:
            raise OnlineError("Growing the live model requires OpenCV and SciPy") from error
        detector = cv2.ORB_create(nfeatures=3500)
        gray = cv2.cvtColor(np.asarray(pixels * 255, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if descriptors is None:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        candidates, colors, scales, support_ids, support_pixels = [], [], [], [], []
        self.last_candidate_support = {}
        pairs = []
        for old_index, old in enumerate(self.images):
            old_pose = old["pose"].numpy()
            if np.linalg.norm(pose[:3, 3] - old_pose[:3, 3]) < .08 * units:
                continue
            diagnostics["baseline_views"] += 1
            old_pixels = old["pixels"].numpy()
            old_gray = cv2.cvtColor(np.asarray(old_pixels * 255, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
            old_keys, old_descriptors = detector.detectAndCompute(old_gray, None)
            if old_descriptors is None:
                continue
            matches = [pair[0] for pair in matcher.knnMatch(descriptors, old_descriptors, k=2)
                       if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
            if len(matches) < 16:
                continue
            pairs.append((old_index, old, old_keys, matches))
        diagnostics["matched_pairs"] = len(pairs)
        # Descriptor count is a cheap overlap proxy. Keep up to twelve older
        # references and all eligible last-four references, with a hard cap.
        strongest = sorted(pairs, key=lambda item: (-len(item[3]), -item[0]))[:12]
        recent = [item for item in pairs if item[0] >= len(self.images) - 4]
        selected_pairs = {item[0]: item for item in strongest + recent}
        for _old_index, old, old_keys, matches in selected_pairs.values():
            uv = np.asarray([keypoints[m.queryIdx].pt for m in matches])
            old_uv = np.asarray([old_keys[m.trainIdx].pt for m in matches])
            # The shared triangulator's depth/baseline thresholds are meters.
            # Convert translations to physical units, keeping rotations and
            # pixel intrinsics unchanged, then return points to model units.
            metric_pose, metric_old = pose.copy(), old["pose"].numpy().copy()
            metric_pose[:3, 3] /= units
            metric_old[:3, 3] /= units
            points, indices = triangulate_correspondences(
                uv, old_uv, metric_pose, metric_old,
                intrinsics, old["intrinsics"].numpy())
            points *= units
            # Six geometrically checked observations can define a useful
            # sparse patch; the old twelve-point gate discarded whole pairs.
            if len(points) < 6:
                continue
            diagnostics["triangulated_pairs"] += 1
            diagnostics["triangulated_points"] += len(points)
            pixel_xy = np.rint(uv[indices]).astype(int)
            pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, pixels.shape[1] - 1)
            pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, pixels.shape[0] - 1)
            depth = np.linalg.norm(points - pose[:3, 3], axis=1)
            radius = np.clip(depth / max(intrinsics[0, 0], intrinsics[1, 1]) * 2., .003 * units, .08 * units)
            candidates.append(points)
            support_ids.extend([old.get("capture_id")] * len(points))
            support_pixels.extend(old_uv[indices])
            colors.append(pixels[pixel_xy[:, 1], pixel_xy[:, 0]])
            scales.append(np.repeat(radius[:, None], 3, axis=1))
        if not candidates:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
        points = np.concatenate(candidates)
        colors = np.concatenate(colors)
        scales = np.concatenate(scales)
        existing = self.params["means"].detach().cpu().numpy()[~self.removed_rows.cpu().numpy()]
        distance = cKDTree(existing).query(points)[0] if len(existing) else np.full(len(points), np.inf)
        selected = np.flatnonzero(distance > .02 * units)
        diagnostics["unoccupied_points"] = len(selected)
        if len(selected):
            # Deduplicate overlapping pairs, then bound both GPU and PLY growth.
            kept = []
            grid = set()
            for index in selected:
                cell = tuple(np.floor(points[index] / (.02 * units)).astype(np.int64))
                if cell not in grid:
                    grid.add(cell)
                    kept.append(index)
                if len(kept) >= min(512, 2_000_000 - len(existing)):
                    break
            selected = np.asarray(kept, dtype=np.int64)
        diagnostics["added_points"] = len(selected)
        if len(selected):
            # A deduplicated anchor may be supported by several matched views.
            # Preserve only references whose independently matched pixel still
            # agrees with that exact retained anchor, not merely a nearby blob.
            nearest_distance, nearest = cKDTree(points[selected]).query(points)
            references = {old.get("capture_id"): old for old in self.images}
            for raw_index, new_index in enumerate(nearest):
                capture_id = support_ids[raw_index]
                old = references.get(capture_id)
                if old is None or not capture_id or nearest_distance[raw_index] > .02 * units:
                    continue
                camera = np.linalg.inv(old["pose"].numpy()) @ np.append(points[selected[new_index]], 1.)
                uv = old["intrinsics"].numpy() @ camera[:3]
                if camera[2] <= .25 * units or np.linalg.norm(uv[:2]/uv[2]-support_pixels[raw_index]) > 2.5:
                    continue
                self.last_candidate_support.setdefault(capture_id, []).append(int(new_index))
            self.last_candidate_support = {cid: sorted(set(indices))
                for cid, indices in self.last_candidate_support.items()}
        return points[selected], colors[selected], scales[selected]

    def _validation_view(self, view: dict) -> bool:
        return bool(view.get("validation_only") or view["capture_id"] in self.candidate_validation_ids)

    def _prepare_candidate_support(self) -> None:
        """Reserve observed overlapping cameras before any candidate gradients."""
        count = len(self.params["means"]) - self.protected_count
        self.candidate_initial_scales = self.torch.exp(
            self.params["log_scales"][self.protected_count:].detach()).cpu().numpy()
        kept = self._last_growth_selected_indices
        remap = {int(old): new for new, old in enumerate(kept)}
        cache = {v["capture_id"]: v for v in self.images}
        self.candidate_observations = {cid: [remap[i] for i in indices if i in remap]
            for cid, indices in self.last_candidate_support.items() if cid in cache}
        self.candidate_observations = {cid: indices for cid, indices in self.candidate_observations.items() if indices}
        self.candidate_validation_ids = []
        remaining = set(range(count))
        while remaining and self.candidate_observations:
            cid = max(self.candidate_observations, key=lambda key:
                len(set(self.candidate_observations[key]) & remaining))
            covered = set(self.candidate_observations[cid]) & remaining
            if not covered:
                break
            self.candidate_validation_ids.append(cid)
            remaining -= covered
        self.candidate_training_ids = [cid for cid in self.candidate_observations
            if cid not in self.candidate_validation_ids and not cache[cid].get("validation_only")]
        # Missing support is recorded rather than invented; review fails closed.
        if remaining:
            self.candidate_validation_ids = []

    def retry_rejected_image(self, capture_id: str, image_path: Path,
                             pose: np.ndarray, intrinsics: np.ndarray) -> None:
        """Reassess one unpublished rejection once per explicit method revision."""
        if (capture_id not in self.rejected_capture_reviews or capture_id not in self.accepted_ids or
                capture_id in self.published_ids or self.has_pending_work or
                self.rejected_capture_reviews[capture_id].get("review_method_revision") == self.review_method_revision or
                self.review_attempts.get(capture_id) == self.review_method_revision):
            raise OnlineError("Rejected photo is not eligible for this bounded review retry")
        self._admit_image(capture_id, image_path, pose, intrinsics)
        self.review_attempts[capture_id] = self.review_method_revision

    @property
    def capture_ids(self) -> list[str]:
        return list(self.accepted_ids)

    @property
    def published_capture_ids(self) -> list[str]:
        return list(self.published_ids)

    @property
    def rejected_capture_ids(self) -> list[str]:
        return list(self.rejected_capture_reviews)

    def _scaled_image(self, image_path: Path,
                      intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use identical pixels and calibration for ingestion and evaluation."""
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            if self.camera_distortion is not None and np.any(self.camera_distortion):
                try:
                    import cv2
                except ImportError as error:
                    raise OnlineError("Measured camera distortion requires OpenCV") from error
                # Preserve native K as the undistorted output calibration.
                # Apply the measured lens model before resizing, identically
                # for baseline matching, new photos and held-out evaluation.
                image = Image.fromarray(cv2.undistort(np.asarray(image),
                    np.asarray(intrinsics, dtype=np.float64), self.camera_distortion))
            shrink = min(1., self.max_image_edge / max(width, height))
            if shrink < 1:
                image = image.resize((max(1, round(width * shrink)), max(1, round(height * shrink))),
                                     Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.
        scaled_k = intrinsics.copy()
        scaled_k[0, :] *= image.width / width
        scaled_k[1, :] *= image.height / height
        return pixels, scaled_k

    def render_l1(self, image_path: Path, pose: np.ndarray,
                  intrinsics: np.ndarray) -> float:
        """Evaluate a held-out camera without changing model or optimizer state."""
        pixels, scaled_k = self._scaled_image(image_path, intrinsics)
        view = {"pixels": self.torch.from_numpy(pixels),
                "pose": self.torch.from_numpy(np.asarray(pose, dtype=np.float32)),
                "intrinsics": self.torch.from_numpy(np.asarray(scaled_k, dtype=np.float32))}
        return self._render_cached_l1(view)

    def _render_cached_l1(self, view: dict, *, published_only: bool = False) -> float:
        return self._render_cached_error(view, published_only=published_only)

    def _render_cached_patch_l1(self, view: dict, mask: np.ndarray, *, published_only: bool = False) -> float:
        return self._render_cached_error(view, published_only=published_only, support_mask=mask)

    def _render_cached_error(self, view: dict, *, published_only: bool = False,
                             support_mask: np.ndarray | None = None) -> float:
        if self._camera_cache_requires_prime:
            raise OnlineError("Re-prime baseline cameras after adopting measured distortion")
        torch = self.torch
        with torch.no_grad():
            image = view["pixels"].to(self.device)
            count = self.protected_count if published_only else len(self.params["means"])
            colors, appearance = self._appearance(count)
            rendered, _alpha, _info = self.rasterization(
                means=self.params["means"][:count],
                quats=torch.nn.functional.normalize(self.params["quats"][:count], dim=-1),
                scales=torch.exp(self.params["log_scales"][:count]).clamp(min=1e-5, max=100.),
                opacities=torch.sigmoid(self.params["opacity_logits"][:count]),
                colors=colors, **appearance,
                viewmats=torch.linalg.inv(view["pose"].to(self.device))[None, :, :],
                Ks=view["intrinsics"].to(self.device)[None, :, :],
                width=image.shape[1], height=image.shape[0])
            error = (rendered[0] - image).abs()
            if support_mask is not None:
                if support_mask.shape != tuple(image.shape[:2]) or not support_mask.any():
                    raise OnlineError("Invalid independently observed candidate footprint")
                error = error[torch.from_numpy(support_mask).to(self.device)]
            loss = error.mean()
            if not torch.isfinite(loss):
                raise OnlineError("Online Gaussian held-out loss became non-finite")
            return float(loss.item())

    def add_image(self, capture_id: str, image_path: Path,
                  pose: np.ndarray, intrinsics: np.ndarray) -> None:
        if capture_id in self.capture_ids and not self.active_replacement:
            return
        self._admit_image(capture_id, image_path, pose, intrinsics)

    def _admit_image(self, capture_id: str, image_path: Path,
                     pose: np.ndarray, intrinsics: np.ndarray) -> None:
        if self._camera_cache_requires_prime:
            raise OnlineError("Re-prime baseline cameras after adopting measured distortion")
        if self.active_replacement:
            if capture_id not in self.active_replacement["capture_ids"]:
                raise OnlineError("Photo was not selected for this replacement area")
            if capture_id in self.active_replacement["used_ids"]:
                return
        if self.has_pending_work and not self.active_replacement and not self.roi_editable_rows.any():
            raise OnlineError("Finish pending candidate refinement before admitting another photo")
        pose, intrinsics = self._validated_view(pose, intrinsics)
        pixels, scaled_k = self._scaled_image(image_path, intrinsics)
        self.last_candidate_support = {}
        points, colors, scales = self._new_points(pixels, pose, scaled_k)
        # A new photograph may not freeze candidates from the preceding
        # photograph before they have actually passed publication review.
        if not self.has_pending_work:
            self._protect_existing()
        added = self._grow(points, colors, scales)
        if added and capture_id not in self.candidate_capture_ids:
            self.candidate_capture_ids.append(capture_id)
        if added and not self.active_replacement and not self.roi_editable_rows.any():
            self._prepare_candidate_support()
        self.images = [v for v in self.images if v["capture_id"] != capture_id]
        self.images.append({"capture_id": capture_id,
                            "new_gaussians": added,
                            "pixels": self.torch.from_numpy(pixels),
                            "pose": self.torch.from_numpy(pose.astype(np.float32)),
                            "intrinsics": self.torch.from_numpy(scaled_k.astype(np.float32))})
        if capture_id not in self.accepted_ids:
            self.accepted_ids.append(capture_id)
        if self.active_replacement:
            self.active_replacement["used_ids"].append(capture_id)
        held_out = [v for v in self.images if self._validation_view(v)]
        training = [v for v in self.images if not self._validation_view(v)]
        self.images = held_out + training[-(self.max_cached_views - len(held_out)):]
        available = {v["capture_id"] for v in self.images}
        self.candidate_training_ids = [cid for cid in self.candidate_training_ids if cid in available]
        self.pending_newest = True

    def train_steps(self, count: int) -> int:
        if self._camera_cache_requires_prime:
            raise OnlineError("Re-prime baseline cameras after adopting measured distortion")
        if type(count) is not int or not 0 <= count <= 100:
            raise OnlineError("Online training step budget exceeds bounds")
        if not self.images:
            return 0
        views = [view for view in self.images if not self._validation_view(view)]
        if self.has_pending_work and self.candidate_observations:
            allowed_ids = set(self.candidate_training_ids + self.candidate_capture_ids)
            views = [view for view in views if view["capture_id"] in allowed_ids]
        if not views:
            return 0
        if self.has_pending_work:
            count = min(count, self.maximum_candidate_steps - self.candidate_steps)
        torch = self.torch
        for _ in range(count):
            # Favor the new overlapping patch while retaining older training
            # cameras. Held-out baseline cameras never receive loss gradients.
            view = views[-1] if self.pending_newest or self.step % 2 == 0 else views[(self.step // 2) % len(views)]
            self.pending_newest = False
            image = view["pixels"].to(self.device)
            height, width = image.shape[:2]
            c2w = view["pose"].to(self.device)
            k = view["intrinsics"].to(self.device)
            viewmat = torch.linalg.inv(c2w)[None, :, :]
            colors, appearance = self._appearance(len(self.params["means"]))
            rendered, _alpha, info = self.rasterization(
                means=self.params["means"],
                quats=torch.nn.functional.normalize(self.params["quats"], dim=-1),
                scales=torch.exp(self.params["log_scales"]).clamp(min=1e-5, max=100.),
                opacities=torch.sigmoid(self.params["opacity_logits"]),
                colors=colors, **appearance,
                viewmats=viewmat, Ks=k[None, :, :], width=width, height=height)
            if self.roi_editable_rows.any():
                if "means2d" not in info:
                    raise OnlineError("ROI refinement requires gsplat projected gradients")
                info["means2d"].retain_grad()
            loss = torch.nn.functional.l1_loss(rendered[0], image)
            if not torch.isfinite(loss):
                raise OnlineError("Online Gaussian loss became non-finite")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.roi_editable_rows.any():
                grads = info["means2d"].grad
                if grads is None:
                    raise OnlineError("ROI refinement has no projected gradients")
                if "gaussian_ids" in info:
                    ids = info["gaussian_ids"].long()
                    valid = info["radii"].max(dim=-1).values > 0
                    grads = grads.reshape(-1, 2)[valid]
                    ids = ids[valid]
                else:
                    valid = (info["radii"] > 0).all(dim=-1)
                    ids = torch.where(valid)[1]
                    grads = grads[valid]
                scores = torch.linalg.vector_norm(grads * torch.tensor(
                    [width / 2., height / 2.], device=grads.device), dim=-1)
                keep = self.roi_editable_rows[ids] & torch.isfinite(scores)
                self.roi_gradient_sum.index_add_(0, ids[keep], scores[keep].detach())
                self.roi_gradient_count.index_add_(
                    0, ids[keep], torch.ones_like(scores[keep]).detach())
            for parameter in self.params.values():
                if parameter.grad is not None:
                    parameter.grad[:self.protected_count][self._frozen_rows] = 0
            self._restore_protected()
            # Candidate footprints may not drift outside the replacement box
            # or into a previously locked volume during optimization.
            constrained = None
            if self.locked_regions or self.active_replacement:
                mutable = self.torch.ones(len(self.params["means"]), dtype=self.torch.bool, device=self.device)
                mutable[:self.protected_count] = ~self._frozen_rows
                ids = self.torch.where(mutable)[0]
                constrained = (ids, {n: p[ids].detach().clone() for n, p in self.params.items()})
            self.optimizer.step()
            if self.roi_editable_rows.any():
                self._bound_reviewed_positions()
            if constrained is not None and len(constrained[0]):
                ids, previous = constrained
                points = self.params["means"][ids].detach().cpu().numpy()
                scales = self.torch.exp(self.params["log_scales"][ids].detach()).cpu().numpy()
                valid = self._allowed_growth(points, scales)
                bad = self.torch.from_numpy(~valid).to(self.device)
                if bad.any():
                    with self.torch.no_grad():
                        for name, param in self.params.items():
                            param[ids[bad]] = previous[name][bad]
                            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                                moment = self.optimizer.state.get(param, {}).get(key)
                                if moment is not None:
                                    moment[ids[bad]] = 0
            self._restore_protected()
            # Keep optimization local to the triangulated physical patch.
            # A quality render cannot authorize unlimited geometry drift.
            if self.has_pending_work and not self.roi_editable_rows.any():
                with torch.no_grad():
                    section = slice(self.protected_count, None)
                    delta = self.params["means"][section] - self.candidate_anchors[section]
                    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                    delta *= (.03 * self.model_units_per_meter / length.clamp(min=1e-8)).clamp(max=1.)
                    self.params["means"][section] = self.candidate_anchors[section] + delta
                    self.params["log_scales"][section].clamp_(
                        min=np.log(.003 * self.model_units_per_meter),
                        max=np.log(.08 * self.model_units_per_meter))
            self.step += 1
            if self.has_pending_work:
                self.candidate_steps += 1
        return count

    def _review_candidates(self) -> dict:
        """Require useful progress on new cameras and no held-out regression."""
        candidates = len(self.params["means"]) - self.protected_count
        report = {"status": "refining", "candidate_gaussians": candidates,
                  "review_method_revision": self.review_method_revision,
                  "training_steps": self.candidate_steps,
                  "minimum_training_steps": self.minimum_candidate_steps}
        if self.candidate_steps < self.minimum_candidate_steps:
            return report
        if self.sh_degree is None:
            return {**report, "status": "unsupported_seed_appearance",
                    "reason": "Seed SH layout cannot be rendered faithfully for quality validation"}
        held_out = [view for view in self.images if self._validation_view(view)]
        targets = [view for view in self.images if view["capture_id"] in self.candidate_capture_ids]
        if len(held_out) < 2 or not targets:
            return {**report, "status": "needs_baseline_validation",
                    "reason": "Two held-out baseline cameras and a new target camera are required"}
        if (not self.candidate_validation_ids or
                self.candidate_initial_scales.shape != (candidates, 3)):
            return {**report, "status": "needs_candidate_support",
                    "reason": "New geometry needs fixed independently observed overlapping validation cameras"}
        cache = {v["capture_id"]: v for v in self.images}
        covered = set()
        for cid in self.candidate_validation_ids:
            if cid not in cache or cid in self.candidate_capture_ids:
                return {**report, "status": "needs_candidate_support"}
            _mask, visible = self._candidate_support_mask(cache[cid])
            covered.update(set(self.candidate_observations.get(cid, [])) & visible)
        if covered != set(range(candidates)):
            return {**report, "status": "needs_candidate_support",
                    "reason": "Every candidate requires a visible independently triangulated validation observation"}
        evaluations = []
        for view in held_out + targets:
            before = self._render_cached_l1(view, published_only=True)
            after = self._render_cached_l1(view)
            tolerance = max(.001, before * .02)
            row = {"capture_id": view["capture_id"],
                                "held_out": self._validation_view(view),
                                "cohort_validation": view["capture_id"] in self.candidate_validation_ids,
                                "before_l1": before, "after_l1": after,
                                "allowed_regression_l1": tolerance}
            mask, _visible = self._candidate_support_mask(view)
            row["support_pixels"] = int(mask.sum())
            if mask.sum() >= 16:
                local_before = self._render_cached_patch_l1(view, mask, published_only=True)
                local_after = self._render_cached_patch_l1(view, mask)
                row.update(support_before_l1=local_before, support_after_l1=local_after,
                    support_improvement_l1=local_before-local_after,
                    support_relative_improvement=(local_before-local_after)/max(local_before, 1e-8),
                    support_allowed_regression_l1=max(.001, local_before*.02))
            evaluations.append(row)
        non_regression = all(row["after_l1"] <= row["before_l1"] + row["allowed_regression_l1"]
                             for row in evaluations)
        target_rows = [row for row in evaluations if not row["held_out"]]
        improvement = float(np.mean([row["before_l1"] - row["after_l1"] for row in target_rows]))
        local_non_regression = all(row.get("support_after_l1", 0.) <=
            row.get("support_before_l1", 0.) + row.get("support_allowed_regression_l1", 0.)
            for row in evaluations if row["held_out"])
        relevant_validation = all(row["support_pixels"] >= 16 for row in evaluations if row["cohort_validation"])
        useful_patch = all(row.get("support_improvement_l1", 0.) > .001 and
            row.get("support_relative_improvement", 0.) >= .02 for row in target_rows)
        passed = non_regression and local_non_regression and relevant_validation and useful_patch and improvement > 0.
        return {**report, "status": "accepted" if passed else "quality_rejected",
                "views": evaluations, "new_view_improvement_l1": improvement,
                "minimum_improvement_l1": 0., "minimum_support_improvement_l1": .001,
                "minimum_support_relative_improvement": .02,
                "non_regression": non_regression, "local_non_regression": local_non_regression,
                "relevant_validation": relevant_validation, "useful_patch": useful_patch,
                "cohort_validation_ids": self.candidate_validation_ids}

    def _candidate_support_mask(self, view: dict) -> tuple[np.ndarray, set[int]]:
        """Project only fixed pre-training anchors/radii; never pick good-error pixels."""
        points = self.candidate_anchors[self.protected_count:].detach().cpu().numpy()
        scales = self.candidate_initial_scales
        h, w = view["pixels"].shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        if scales.shape != points.shape:
            return mask, set()
        pose, k = view["pose"].numpy(), view["intrinsics"].numpy()
        camera = (np.linalg.inv(pose) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
        projected = (k @ camera.T).T
        visible = set()
        for index, (point, pixel, scale) in enumerate(zip(camera, projected, scales)):
            if point[2] <= .25*self.model_units_per_meter or not np.isfinite(pixel).all():
                continue
            uv = pixel[:2]/pixel[2]
            if not (0 <= uv[0] < w and 0 <= uv[1] < h):
                continue
            radius = min(64., max(2., 3.*max(scale)*max(k[0, 0], k[1, 1])/point[2]))
            x0, x1 = max(0, int(np.floor(uv[0]-radius))), min(w, int(np.ceil(uv[0]+radius+1)))
            y0, y1 = max(0, int(np.floor(uv[1]-radius))), min(h, int(np.ceil(uv[1]+radius+1)))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] |= (xx-uv[0])**2+(yy-uv[1])**2 <= radius*radius
            visible.add(index)
        return mask, visible

    def _clear_candidate_support(self):
        self.candidate_validation_ids = []
        self.candidate_training_ids = []
        self.candidate_observations = {}
        self.candidate_initial_scales = np.empty((0, 3), dtype=np.float32)

    def _discard_candidates(self):
        """Drop only unpublished rows, keeping old identities and Adam state."""
        torch = self.torch
        keep = self.protected_count
        previous, old_optimizer = self.params, self.optimizer
        params = torch.nn.ParameterDict({name: torch.nn.Parameter(value[:keep].detach().clone())
                                        for name, value in previous.items()})
        optimizer = torch.optim.Adam(params.parameters(), lr=old_optimizer.param_groups[0]["lr"])
        for name, old in previous.items():
            state = old_optimizer.state.get(old)
            if state:
                optimizer.state[params[name]] = {
                    key: value[:keep].clone() if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} else
                    value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in state.items()}
        self.params, self.optimizer = params, optimizer
        self.candidate_anchors = self.candidate_anchors[:keep].clone()
        for name in ("roi_editable_rows", "removed_rows", "roi_gradient_sum", "roi_gradient_count"):
            setattr(self, name, getattr(self, name)[:keep].clone())
        for capture_id in self.candidate_capture_ids:
            self.rejected_capture_reviews[capture_id] = dict(self.last_publication_review)
        self.candidate_capture_ids.clear()
        self._clear_candidate_support()
        self.candidate_steps = 0
        self.pending_newest = False
        self._protect_existing()

    def save(self, checkpoint: Path, ply: Path) -> dict:
        if self._camera_cache_requires_prime:
            raise OnlineError("Re-prime baseline cameras after adopting measured distortion")
        torch = self.torch
        self._restore_protected()
        self._validate_reviewed_anchor_bound()
        publish_candidates = not self.has_pending_work
        if self.has_pending_work and not self.roi_editable_rows.any():
            self.last_publication_review = self._review_candidates()
            publish_candidates = self.last_publication_review["status"] == "accepted"
            if not publish_candidates and self.candidate_steps >= self.maximum_candidate_steps:
                self._discard_candidates()
                self.last_publication_review["status"] = "discarded"
        elif self.roi_editable_rows.any():
            # Explicit user-reviewed area edits retain their existing owner;
            # normal capture never enables this mutable lineage.
            publish_candidates = True
        with torch.no_grad():
            if not publish_candidates:
                vertices = self._published_vertex.copy()
            else:
                vertices = self._export_vertices()
                self.added_total += max(0, len(vertices) - len(self._published_vertex))
                self.published_ids.extend(capture_id for capture_id in self.candidate_capture_ids
                                          if capture_id not in self.published_ids)
                for capture_id in self.candidate_capture_ids:
                    self.rejected_capture_reviews.pop(capture_id, None)
                self.candidate_capture_ids.clear()
                self._clear_candidate_support()
                self._published_vertex = vertices.copy()
                self._protect_existing()
                self._cache_published_appearance()
                self.candidate_steps = 0
            self._write_published_ply(vertices, ply)
        payload = {"schema_version": 4, "seed_count": self.seed_count,
                   "parameters": self.params.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "images": self.images, "accepted_ids": self.accepted_ids,
                   "published_capture_ids": self.published_ids,
                   "candidate_capture_ids": self.candidate_capture_ids,
                   "candidate_observations": self.candidate_observations,
                   "candidate_validation_ids": self.candidate_validation_ids,
                   "candidate_training_ids": self.candidate_training_ids,
                   "candidate_initial_scales": self.candidate_initial_scales,
                   "review_attempts": self.review_attempts,
                   "rejected_capture_reviews": self.rejected_capture_reviews,
                   "added_total": self.added_total,
                   "roi_editable_rows": self.roi_editable_rows.detach().cpu(),
                   "roi_anchor_guard_version": 1,
                   "roi_children_total": self.roi_children_total,
                   "removed_rows": self.removed_rows.detach().cpu(),
                   "locked_regions": self.locked_regions,
                   "active_replacement": self.active_replacement,
                   "roi_gradient_sum": self.roi_gradient_sum.detach().cpu(),
                   "roi_gradient_count": self.roi_gradient_count.detach().cpu(),
                   "pending_newest": self.pending_newest,
                   "published_count": len(self._published_vertex),
                   "model_units_per_meter": self.model_units_per_meter,
                   "camera_distortion": (self.camera_distortion.tolist()
                                         if self.camera_distortion is not None else None),
                   "candidate_steps": self.candidate_steps,
                   "candidate_anchors": self.candidate_anchors.detach().cpu(),
                   "publication_review": self.last_publication_review,
                   "step": self.step}
        torch.save(payload, checkpoint)
        return {"training_steps": self.step, "accepted_views": len(self.accepted_ids),
                "processed_views": len(self.accepted_ids),
                "published_views": len(self.published_ids),
                "pending_views": len(self.candidate_capture_ids),
                "rejected_views": len(self.rejected_capture_reviews),
                "gaussians": len(self._published_vertex), "optimizer_gaussians": len(self.params["means"]),
                "new_gaussians": self.added_total, "pending_refinement": self.pending_refinement,
                "published_capture_ids": self.published_capture_ids,
                "rejected_capture_ids": self.rejected_capture_ids,
                "rejected_capture_reviews": self.rejected_capture_reviews}

    def _write_published_ply(self, vertices, destination: Path):
        """Keep the exact reviewed file when its complete Gaussian records are unchanged.

        Re-serializing a frozen seed can strip header metadata without changing
        geometry. That invented a new full-model identity while producing the
        same reduced display, conflicting with its immutable source receipt.
        Compare every property, including all SH coefficients, before reusing
        bytes. A real geometry/appearance change still gets a new export.
        """
        from plyfile import PlyData, PlyElement
        source = self._published_ply_path
        if source.resolve() == destination.resolve():
            raise OnlineError("Published Gaussian export needs a separate staging path")
        original = PlyData.read(str(source))["vertex"].data
        if (original.dtype == vertices.dtype and original.shape == vertices.shape and
                original.tobytes() == vertices.tobytes()):
            shutil.copyfile(source, destination)
        else:
            PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(destination))

    def _export_vertices(self):
        torch = self.torch
        with torch.no_grad():
            p = {name: value.detach().float().cpu().numpy() for name, value in self.params.items()}
            quats = p["quats"] / np.maximum(np.linalg.norm(p["quats"], axis=1, keepdims=True), 1e-8)
            rgb = 1 / (1 + np.exp(-np.clip(p["color_logits"], -30, 30)))
            dc = (rgb - 0.5) / SH_C0
            # Keep every frozen published record byte-for-byte, including
            # higher-order SH and unknown source properties. New points start
            # with zero for unsupported extra coefficients.
            vertices = np.zeros(len(p["means"]), dtype=self._published_vertex.dtype)
            for names, values in ((["x", "y", "z"], p["means"]),
                                  (["f_dc_0", "f_dc_1", "f_dc_2"], dc),
                                  (["scale_0", "scale_1", "scale_2"], p["log_scales"]),
                                  (["rot_0", "rot_1", "rot_2", "rot_3"], quats)):
                for index, name in enumerate(names):
                    vertices[name] = values[:, index]
            vertices["opacity"] = p["opacity_logits"]
            # Explicitly editable rows retain their original non-core fields.
            # Every other already-published row is copied as a complete record.
            previous_count = min(len(self._published_vertex), len(vertices))
            editable = self.roi_editable_rows[:previous_count].detach().cpu().numpy()
            edited_core = vertices[:previous_count][editable].copy()
            vertices[:previous_count] = self._published_vertex[:previous_count]
            for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
                         "rot_2", "rot_3", "opacity"):
                vertices[name][:previous_count][editable] = edited_core[name]
        return vertices
