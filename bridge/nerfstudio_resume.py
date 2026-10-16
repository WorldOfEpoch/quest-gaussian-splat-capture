"""Run Nerfstudio with pinned Gaussian optimizer and opacity-reset repairs.

The installed Nerfstudio trainer constructs Adam before loading the checkpoint.
SplatfactoModel.load_state_dict then replaces each Gaussian Parameter, leaving
Adam attached to the abandoned Parameters. This narrow entrypoint runs the
normal ns-train CLI and repairs those references immediately after load. The
installed gsplat DefaultStrategy also has a precedence bug that makes its
positive-step opacity reset unreachable; patch only that exact known source in
this training process, leaving the installed dependency untouched.
"""

from __future__ import annotations

import hashlib
import inspect
import json


GAUSSIAN_GROUPS = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
RESUME_RUNNER_VERSION = 3
OPACITY_RESET_COMPATIBILITY = "gsplat_default_reset_v2"
OPACITY_RESET_MARKER = "Verified gsplat DefaultStrategy opacity reset compatibility: "
RESUME_WARMUP_MARKER = "Verified resumed Splatfacto refinement warmup: "
# SHA256 of inspect.getsource(DefaultStrategy.step_post_backward) in the
# installed gsplat runtime. The corrected hash differs only in the reset if.
_FAULTY_STRATEGY_SHA256 = "5209fb9c548d6e1188ae001eefaa6db4df9581591e2cddd60e0500c8c6066905"
_CORRECTED_STRATEGY_SHA256 = "f99dc8b1d1d583adc1700ab50bc45a7d648d511fa9f1711f93605230bd9540eb"


def install_opacity_reset_compatibility(strategy_cls, reset_opacity) -> str:
    """Patch only the pinned faulty method; already-corrected source is left alone."""
    original = strategy_cls.step_post_backward
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as error:
        raise RuntimeError("Cannot verify gsplat opacity reset implementation") from error
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest == _CORRECTED_STRATEGY_SHA256:
        return "already_corrected"
    if digest != _FAULTY_STRATEGY_SHA256:
        raise RuntimeError(f"Unknown gsplat opacity reset implementation: {digest}")

    def repaired(self, params, optimizers, state, step, info, packed=False):
        result = original(self, params, optimizers, state, step, info, packed=packed)
        # Mirror the intended final statement of the verified source, after
        # update/grow/prune. The original's `0 & step > 0` never evaluates true.
        if 0 < step < self.refine_stop_iter and self.reset_every > 0 and step % self.reset_every == 0:
            reset_opacity(params=params, optimizers=optimizers, state=state,
                          value=self.prune_opa * 2.0)
        return result

    strategy_cls.step_post_backward = repaired
    return "patched_faulty"


def rebind_loaded_gaussians(trainer) -> None:
    if trainer._start_step <= 0:
        raise RuntimeError("Resume runner did not load a checkpoint")
    model = trainer.pipeline.model
    for name in GAUSSIAN_GROUPS:
        restored = model.gauss_params[name]
        optimizer = trainer.optimizers.optimizers[name]
        original = trainer.optimizers.parameters[name]
        if (len(original) != 1 or len(optimizer.param_groups) != 1 or
                len(optimizer.param_groups[0]["params"]) != 1 or
                optimizer.param_groups[0]["params"][0] is not original[0] or
                original[0] is restored):
            raise RuntimeError(f"Unexpected resumed Gaussian optimizer binding: {name}")
        state = optimizer.state.pop(original[0], None)
        if (not state or "exp_avg" not in state or "exp_avg_sq" not in state or
                tuple(state["exp_avg"].shape) != tuple(restored.shape) or
                tuple(state["exp_avg_sq"].shape) != tuple(restored.shape) or
                restored in optimizer.state):
            raise RuntimeError(f"Unexpected resumed Gaussian optimizer state: {name}")
        optimizer.param_groups[0]["params"][0] = restored
        optimizer.state[restored] = state
        trainer.optimizers.parameters[name] = [restored]
    print("Verified Splatfacto optimizer bindings for six restored Gaussian parameter groups", flush=True)


def configure_resumed_refinement_warmup(trainer) -> dict:
    """Accumulate a full camera cycle before a resumed growth/prune decision.

    Splatfacto does not save gsplat's running refinement statistics. After a
    checkpoint load, DefaultStrategy starts with empty grad/count arrays while
    the absolute step may already land on a refine boundary. Raising only its
    `refine_start_iter` leaves `_update_state` and the independent opacity-reset
    schedule untouched. FullImageDatamanager's initial unseen list is one
    permutation of all training cameras; all of it must be visited first.
    """
    if trainer._start_step <= 0:
        raise RuntimeError("Refinement warmup requires a loaded checkpoint")
    model = trainer.pipeline.model
    manager = trainer.pipeline.datamanager
    strategy = model.strategy
    state = model.strategy_state
    cameras = list(manager.train_unseen_cameras)
    count = len(manager.train_dataset)
    if (type(strategy).__module__ != "gsplat.strategy.default" or
            type(strategy).__name__ != "DefaultStrategy" or
            type(manager).__module__ != "nerfstudio.data.datamanagers.full_images_datamanager" or
            type(manager).__name__ != "FullImageDatamanager" or
            not isinstance(state, dict) or
            state.get("grad2d") is not None or state.get("count") is not None or
            ("radii" in state and state["radii"] is not None) or
            not isinstance(count, int) or count <= 0 or
            count != model.num_train_data or
            len(cameras) != count or sorted(cameras) != list(range(count)) or
            not isinstance(strategy.refine_every, int) or strategy.refine_every <= 0 or
            not isinstance(strategy.refine_stop_iter, int) or strategy.refine_stop_iter <= 0 or
            not isinstance(strategy.refine_start_iter, int) or strategy.refine_start_iter < 0 or
            not isinstance(strategy.reset_every, int) or strategy.reset_every <= 0):
        raise RuntimeError("Cannot verify resumed Splatfacto camera cycle/refinement strategy")
    start = trainer._start_step
    warmup = max(count, strategy.refine_every)
    through = start + warmup - 1
    strategy.refine_start_iter = max(strategy.refine_start_iter, through)
    evidence = {"start_step": start, "train_camera_count": count,
                "refine_every": strategy.refine_every, "warmup_steps": warmup,
                "growth_hold_through_step": through,
                "effective_refine_start_iter": strategy.refine_start_iter,
                "refinement_available": start < strategy.refine_stop_iter}
    print(RESUME_WARMUP_MARKER + json.dumps(evidence, sort_keys=True,
                                          separators=(",", ":")), flush=True)
    return evidence


def main() -> None:
    from gsplat.strategy.default import DefaultStrategy
    from gsplat.strategy.ops import reset_opa
    from nerfstudio.engine.trainer import Trainer
    from nerfstudio.scripts.train import entrypoint

    compatibility = install_opacity_reset_compatibility(DefaultStrategy, reset_opa)
    print(OPACITY_RESET_MARKER + compatibility, flush=True)
    original_load = Trainer._load_checkpoint

    def load_and_rebind(self):
        original_load(self)
        if self._start_step > 0:
            rebind_loaded_gaussians(self)
            configure_resumed_refinement_warmup(self)
        else:
            print("Fresh Splatfacto training; no checkpoint optimizer rebinding needed", flush=True)

    Trainer._load_checkpoint = load_and_rebind
    entrypoint()


if __name__ == "__main__":
    main()
