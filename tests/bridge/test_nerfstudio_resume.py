from types import SimpleNamespace
import hashlib
import inspect
import os
import subprocess
import sys

import pytest

from bridge.nerfstudio_resume import (GAUSSIAN_GROUPS,
    RESUME_RUNNER_VERSION, configure_resumed_refinement_warmup,
    install_opacity_reset_compatibility, rebind_loaded_gaussians)


class FakeParameter:
    def __init__(self, shape):
        self.shape = shape


def _trainer():
    restored = {}
    optimizers = {}
    parameters = {}
    former = {}
    for name in GAUSSIAN_GROUPS:
        old = FakeParameter((7, 3))
        new = FakeParameter((7, 3))
        former[name] = old
        restored[name] = new
        parameters[name] = [old]
        optimizers[name] = SimpleNamespace(
            param_groups=[{"params": [old]}],
            state={old: {"step": 5000, "exp_avg": FakeParameter((7, 3)),
                         "exp_avg_sq": FakeParameter((7, 3))}})
    trainer = SimpleNamespace(
        _start_step=5000,
        pipeline=SimpleNamespace(model=SimpleNamespace(gauss_params=restored)),
        optimizers=SimpleNamespace(optimizers=optimizers, parameters=parameters))
    return trainer, former


def test_loaded_optimizer_state_follows_restored_gaussian_parameters():
    trainer, former = _trainer()
    rebind_loaded_gaussians(trainer)
    for name in GAUSSIAN_GROUPS:
        optimizer = trainer.optimizers.optimizers[name]
        restored = trainer.pipeline.model.gauss_params[name]
        assert optimizer.param_groups[0]["params"][0] is restored
        assert trainer.optimizers.parameters[name][0] is restored
        assert optimizer.state[restored]["step"] == 5000
        assert former[name] not in optimizer.state


def test_resume_rejects_state_with_wrong_gaussian_shape():
    trainer, _ = _trainer()
    trainer.optimizers.optimizers["means"].state[
        trainer.optimizers.parameters["means"][0]]["exp_avg"] = FakeParameter((6, 3))
    with pytest.raises(RuntimeError, match="optimizer state: means"):
        rebind_loaded_gaussians(trainer)


def test_opacity_patch_leaves_exact_corrected_upstream_method_untouched(monkeypatch):
    class Strategy:
        def step_post_backward(self, *_args, **_kwargs):
            return "upstream"

    original = Strategy.step_post_backward
    digest = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    monkeypatch.setattr("bridge.nerfstudio_resume._CORRECTED_STRATEGY_SHA256", digest)
    assert install_opacity_reset_compatibility(Strategy, lambda **_: pytest.fail("double reset")) == "already_corrected"
    assert Strategy.step_post_backward is original
    assert Strategy().step_post_backward() == "upstream"


def test_opacity_patch_rejects_unknown_upstream_method():
    class Strategy:
        def step_post_backward(self, *_args, **_kwargs):
            return "unreviewed"

    original = Strategy.step_post_backward
    with pytest.raises(RuntimeError, match="Unknown gsplat opacity reset implementation"):
        install_opacity_reset_compatibility(Strategy, lambda **_: None)
    assert Strategy.step_post_backward is original


def _warmup_trainer(*, start=5000, views=117, refine=100, stop=15000,
                    initial_cameras=None):
    Strategy = type("DefaultStrategy", (), {
        "__module__": "gsplat.strategy.default"})
    Manager = type("FullImageDatamanager", (), {
        "__module__": "nerfstudio.data.datamanagers.full_images_datamanager"})
    strategy = Strategy()
    strategy.refine_every = refine
    strategy.refine_start_iter = 500
    strategy.refine_stop_iter = stop
    strategy.reset_every = 6000
    manager = Manager()
    manager.train_dataset = range(views)
    manager.train_unseen_cameras = (list(range(views)) if initial_cameras is None
                                    else initial_cameras)
    trainer = SimpleNamespace(_start_step=start, pipeline=SimpleNamespace(
        model=SimpleNamespace(strategy=strategy, strategy_state={"grad2d": None,
            "count": None, "radii": None}, num_train_data=views),
        datamanager=manager))
    return trainer


def test_resumed_warmup_requires_complete_camera_cycle_and_refine_interval(capsys):
    assert RESUME_RUNNER_VERSION == 3
    trainer = _warmup_trainer()
    evidence = configure_resumed_refinement_warmup(trainer)
    assert evidence == {"start_step": 5000, "train_camera_count": 117,
        "refine_every": 100, "warmup_steps": 117,
        "growth_hold_through_step": 5116, "effective_refine_start_iter": 5116,
        "refinement_available": True}
    assert trainer.pipeline.model.strategy.refine_start_iter == 5116
    assert "Verified resumed Splatfacto refinement warmup:" in capsys.readouterr().out


def test_resumed_warmup_keeps_reset_schedule_and_stopped_refinement():
    trainer = _warmup_trainer(start=15000, views=53, refine=100, stop=15000)
    previous_reset = trainer.pipeline.model.strategy.reset_every
    evidence = configure_resumed_refinement_warmup(trainer)
    assert evidence["warmup_steps"] == 100
    assert evidence["growth_hold_through_step"] == 15099
    assert evidence["refinement_available"] is False
    assert trainer.pipeline.model.strategy.reset_every == previous_reset


def test_resumed_warmup_rejects_partial_or_duplicate_camera_cycle():
    for cameras in (list(range(116)), list(range(116)) + [0]):
        trainer = _warmup_trainer(initial_cameras=cameras)
        with pytest.raises(RuntimeError, match="camera cycle"):
            configure_resumed_refinement_warmup(trainer)


def test_resumed_warmup_rejects_restored_strategy_statistics():
    trainer = _warmup_trainer()
    trainer.pipeline.model.strategy_state["count"] = 1
    with pytest.raises(RuntimeError, match="camera cycle"):
        configure_resumed_refinement_warmup(trainer)


def test_installed_strategy_accumulates_before_growth_and_preserves_opacity_reset():
    """Opt-in integration check against the installed Nerfstudio/gsplat Python."""
    runtime = os.getenv("QUEST_NERFSTUDIO_PYTHON")
    if not runtime:
        pytest.skip("Set QUEST_NERFSTUDIO_PYTHON to verify the installed GPU trainer")
    code = r'''
import json
import torch
from gsplat.strategy.default import DefaultStrategy
from bridge.nerfstudio_resume import install_opacity_reset_compatibility
events=[]
assert install_opacity_reset_compatibility(DefaultStrategy,
    lambda **kw: events.append(('reset', kw['step'] if 'step' in kw else 6000))) == 'patched_faulty'
strategy=DefaultStrategy(refine_every=100, reset_every=6000,
    refine_start_iter=5116, refine_stop_iter=15000, pause_refine_after_reset=0)
state={'grad2d':torch.zeros(1),'count':torch.zeros(1)}
def update(params,state,info,packed=False):
    state['grad2d'] += 1
    state['count'] += 1
def grow(params,optimizers,state,step):
    events.append(('grow',step,int(state['count'][0])))
    return 0,0
def prune(params,optimizers,state,step):
    events.append(('prune',step))
    return 0
strategy._update_state=update
strategy._grow_gs=grow
strategy._prune_gs=prune
params={'means':torch.zeros(1,3)}
for step in range(5000,5201):
    strategy.step_post_backward(params,{},state,step,{})
assert [x for x in events if x[0]=='grow']==[('grow',5200,201)], events
assert int(state['count'][0])==0
for step in range(5201,6001):
    strategy.step_post_backward(params,{},state,step,{})
assert len([x for x in events if x[0]=='reset'])==1, events
print(json.dumps({'first_growth_step':5200,'accumulated_views':201,
                  'opacity_resets':1,'post_growth_state_reset':True}))
'''
    result = subprocess.run([runtime, "-c", code], cwd=os.getcwd(),
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '"first_growth_step": 5200' in result.stdout
