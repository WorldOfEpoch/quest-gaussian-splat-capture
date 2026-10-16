"""CUDA area lifecycle regression, isolated synthetic geometry only."""
import uuid
import numpy as np
import pytest
from PIL import Image


def test_protect_existing_area_is_unchanged_and_blocks_overlapping_growth_after_resume(tmp_path):
    torch = pytest.importorskip('torch')
    plyfile = pytest.importorskip('plyfile')
    pytest.importorskip('gsplat')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from bridge.online_torch import GsplatTrainer
    from bridge.online import OnlineError
    names = ('x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')
    vertices = np.zeros(3, dtype=[(name, '<f4') for name in names])
    vertices['x'] = [0., 1., 2.]; vertices['z'] = 2.
    for axis in range(3):
        vertices[f'scale_{axis}'] = -4.
    vertices['rot_0'] = 1.
    seed = tmp_path / 'source.ply'
    plyfile.PlyData([plyfile.PlyElement.describe(vertices, 'vertex')], text=False).write(str(seed))
    trainer = GsplatTrainer(seed, max_image_edge=128)
    trainer.accepted_ids = ['original-photo']; trainer.published_ids = ['original-photo']
    before = {name: value.detach().clone() for name, value in trainer.params.items()}
    region = {'id': str(uuid.uuid4()), 'name': 'Good cabinet',
              'min': [-.2, -.2, 1.7], 'max': [.2, .2, 2.3]}
    assert trainer.protect_existing_area(region) == region
    ckpt, ply = tmp_path / 'locked.ckpt', tmp_path / 'locked.ply'
    metrics = trainer.save(ckpt, ply)
    assert ply.read_bytes() == seed.read_bytes()
    assert metrics['training_steps'] == 0 and metrics['new_gaussians'] == 0
    assert not trainer.removed_rows.any()
    for name, value in trainer.params.items():
        assert torch.equal(value, before[name])
    resumed = GsplatTrainer(seed, ckpt, published_ply=ply, max_image_edge=128)
    assert resumed.locked_regions == [region]
    assert resumed.capture_ids == ['original-photo']
    assert resumed._grow(np.array([[0., 0., 2.], [.21, 0., 2.], [3., 0., 2.]]),
        np.full((3, 3), .5), np.full((3, 3), .01)) == 1
    with pytest.raises(OnlineError, match='refinement'):
        resumed.protect_existing_area({**region, 'id': str(uuid.uuid4()), 'min': [4., 4., 4.], 'max': [5., 5., 5.]})
    with pytest.raises(OnlineError, match='locked'):
        trainer.begin_area_replacement(region, [])


def test_replace_lock_and_resume_preserve_outside_and_block_new_overlap(tmp_path):
    torch = pytest.importorskip('torch')
    plyfile = pytest.importorskip('plyfile')
    pytest.importorskip('gsplat')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from bridge.online_torch import GsplatTrainer
    from bridge.online import OnlineError
    names = ('x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2',
             'rot_0','rot_1','rot_2','rot_3','f_rest_0')
    vertices = np.zeros(4, dtype=[(name, '<f4') for name in names])
    vertices['x'] = [0., 1., -1., 1.]
    vertices['z'] = 2.
    vertices['scale_0'] = vertices['scale_1'] = vertices['scale_2'] = -4.
    vertices['rot_0'] = 1.
    vertices['f_rest_0'] = [.1,.2,.3,.4]
    seed = tmp_path / 'seed.ply'
    plyfile.PlyData([plyfile.PlyElement.describe(vertices, 'vertex')], text=False).write(str(seed))
    original_bytes = seed.read_bytes()
    trainer = GsplatTrainer(seed, max_image_edge=128)
    trainer.accepted_ids = ['replayed']
    box = {'id': str(uuid.uuid4()), 'name': 'Countertop', 'min': [-.2,-.2,1.7], 'max': [.2,.2,2.3]}
    assert trainer.begin_area_replacement(box, ['replayed','second'])['removed_gaussians'] == 1
    assert trainer.params['opacity_logits'][0].item() == -30.
    assert trainer.removed_rows.tolist() == [True,False,False,False]
    photo = tmp_path / 'photo.png'
    Image.new('RGB', (128,128), (200,100,20)).save(photo)
    k = np.array([[100.,0.,64.],[0.,100.,64.],[0.,0.,1.]])
    trainer._new_points = lambda *args: (np.array([[0.,0.,2.],[1.,0.,2.]]),
                                       np.full((2,3),.5), np.full((2,3),.01))
    trainer.add_image('replayed', photo, np.eye(4), k)
    assert len(trainer.params['means']) == 5
    assert trainer.capture_ids == ['replayed']
    with pytest.raises(OnlineError, match='not selected'):
        trainer.add_image('unapproved', photo, np.eye(4), k)
    trainer.add_image('replayed', photo, np.eye(4), k)
    assert len(trainer.params['means']) == 5
    # Use the real CUDA rasterizer for training before and after the first save.
    trainer.train_steps(2)
    ckpt, ply = tmp_path/'candidate.ckpt', tmp_path/'current.ply'
    trainer.save(ckpt, ply)
    trainer.train_steps(2)
    expected = {name: value.detach().clone() for name,value in trainer.params.items()}
    trainer.lock_replacement_area()
    trainer.save(ckpt, ply)
    saved = plyfile.PlyData.read(str(ply))['vertex'].data
    assert saved[1:4].tobytes() == vertices[1:4].tobytes()
    np.testing.assert_allclose(saved['x'], expected['means'][:,0].cpu().numpy(), atol=1e-6)
    resumed = GsplatTrainer(seed, ckpt, published_ply=ply, max_image_edge=128)
    assert resumed.locked_regions == [box]
    assert resumed.removed_rows.tolist() == [True,False,False,False,False]
    for name,value in resumed.params.items():
        assert torch.equal(value, expected[name]), name
    # Growth with its center outside the box but footprint reaching inside
    # must also be rejected. A genuinely separate point can still be added.
    assert resumed._grow(np.array([[0.,0.,2.],[.21,0.,2.],[2.,0.,2.]]),
                         np.full((3,3),.5),np.full((3,3),.01)) == 1
    assert len(resumed.params['means']) == 6
    # The minimum stored scale expands this tiny candidate into the box.
    assert resumed._grow(np.array([[.205,0.,2.]]),np.full((1,3),.5),
                         np.full((1,3),.0001)) == 0
    resumed._protect_existing()
    resumed.train_steps(2)
    for name,value in resumed.params.items():
        assert torch.equal(value[:5], expected[name]), name
    with pytest.raises(OnlineError, match='locked'):
        resumed.begin_area_replacement({**box,'id':str(uuid.uuid4())}, [])
    resumed.unlock_area(box['id'])
    assert not resumed.locked_regions
    assert seed.read_bytes() == original_bytes
