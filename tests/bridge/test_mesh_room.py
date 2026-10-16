"""Physical geometry and evidence gates for the declared architectural prior."""
import numpy as np
import pytest
import uuid
pytest.importorskip('scipy', reason='Optional architectural mesh dependency is unavailable')
from bridge.mesh_room import plane_grids, add_opening_evidence, build_room_surface, select_room_planes


def test_rectangular_room_has_straight_walls_perpendicular_corners_without_equal_lengths():
    bounds=np.array([[-2.,-3.,0.],[2.,3.,2.8]])
    basis=np.array([[0,1,0],[1,0,0],[0,0,1.]])  # Reflecting a camera frame must not invert faces.
    grids=plane_grids(bounds,.2)
    points=np.concatenate([g['vertices'] for g in grids])@basis.T
    v,f,c,report=build_room_surface(points,np.ones_like(points)*.5,basis,bounds,grids)
    local=v@basis
    assert report['spans_reconstruction_units'] == [4,6,2.8]
    assert report['corner_angles_degrees'] == [90,90,90]
    assert 'not generated' in report['ceiling']
    center=bounds.mean(0)@basis.T
    t=v[f]; n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    assert np.all(np.sum(n*(center-t.mean(1)),axis=1)>0)
    for triangle in local[f]:
        assert np.any(np.ptp(triangle,axis=0)<1e-10)


def test_opening_needs_multiple_camera_depths_not_repeated_pixels_or_unknown_depth():
    bounds=np.array([[-2.,-3.,0.],[2.,3.,3.]])
    grids=plane_grids(bounds,.2)
    yy,zz=np.meshgrid(np.linspace(-.8,.8,25),np.linspace(.8,2.2,25))
    directions=np.column_stack((np.full(yy.size,2.),yy.ravel(),zz.ravel()-1.5))
    origin=np.array([0,0,1.5])
    add_opening_evidence(grids,bounds,origin,directions,np.full(yy.size,2.),.2)
    assert grids[1]['votes'].max()==1
    points=np.concatenate([g['vertices'] for g in grids]); colors=np.ones_like(points)*.5
    _,f,_,report=build_room_surface(points,colors,np.eye(3),bounds,grids)
    assert sum(s['depth_proven_opening_cells'] for s in report['surfaces'])==0
    count=len(f)
    add_opening_evidence(grids,bounds,origin,directions,np.zeros(yy.size),.2)
    add_opening_evidence(grids,bounds,origin,directions,np.full(yy.size,np.nan),.2)
    assert grids[1]['votes'].max()==1
    for _ in range(2): add_opening_evidence(grids,bounds,origin,directions,np.full(yy.size,2.),.2)
    _,f,_,report=build_room_surface(points,colors,np.eye(3),bounds,grids)
    assert len(f)<count
    assert report['surfaces'][1]['depth_proven_opening_cells']>8
    assert report['surfaces'][4]['depth_proven_opening_cells']==0


def test_invalid_unbounded_architecture_is_rejected():
    with pytest.raises(ValueError): plane_grids(np.array([[0.,0,0],[1,1,1]]),1e-6)
    with pytest.raises(ValueError): plane_grids(np.array([[0.,0,0],[1,1,1]]),0)
    with pytest.raises(ValueError): select_room_planes(np.ones((1000,3)),np.ones((1000,3)),[],np.eye(3),np.zeros((3,3)),1,1)


def test_texture_view_budget_is_rejected_before_reading_or_mutating_a_scan():
    pytest.importorskip('pymeshlab', reason='Optional photo texture dependency is unavailable')
    from bridge.mesh import finish_mesh
    ids=[str(uuid.uuid4()) for _ in range(3)]
    with pytest.raises(RuntimeError,match='texture view selection budget'):
        finish_mesh(None,*ids,texture_min_cos_angle=.9)
    with pytest.raises(RuntimeError,match='texture view selection budget'):
        finish_mesh(None,*ids,texture_smoothing_iterations=1000)
