import numpy as np
import pytest

from bridge.online import OnlineError
from bridge.region_edit import review_depth_conflicts


def fixture():
    points=np.array([[0.,0.,2.],[0.,0.,4.],[0.,0.,6.]])
    scales=np.full((3,3),.015)
    rotations=np.broadcast_to(np.eye(3),(3,3,3)).copy()
    views=[]
    for i,x in enumerate((-.2,0.,.2)):
        w=np.eye(4);w[0,3]=-x
        views.append(dict(capture_id=str(i),world_to_camera=w,
            intrinsics=np.array([[80.,0,50.],[0,80.,50.],[0,0,1.]]),
            camera_center_m=np.array([x,0,0]),depth=np.full((101,101),4.)))
    return points,scales,rotations,views


def test_only_full_footprint_in_front_of_observed_surfaces_is_candidate():
    result=review_depth_conflicts(*fixture(),clearance=.04)
    assert result['candidate_mask'].tolist()==[True,False,False]
    assert result['surface_supported'].tolist()==[False,True,False]
    assert result['free_view_count'].tolist()==[3,0,0]


def test_depth_holes_are_unknown_even_when_center_pixel_has_depth():
    p,s,r,v=fixture()
    v[0]['depth'][51,58]=0
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'].any()


@pytest.mark.parametrize('bad',[0.,np.nan,np.inf,-1.])
def test_invalid_depth_never_votes_for_free_space(bad):
    p,s,r,v=fixture();v[0]['depth'][:]=bad
    result=review_depth_conflicts(p,s,r,v,clearance=.04)
    assert not result['candidate_mask'].any()
    assert not result['free_view_mask'][:,0].any()


def test_one_surface_agreement_vetoes_other_free_votes():
    p,s,r,v=fixture()
    w=np.eye(4);w[0,3]=-.4
    v.append({**v[0],'world_to_camera':w,'capture_id':'surface','camera_center_m':np.array([.4,0,0]),'depth':np.full((101,101),2.)})
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'][0]


def test_repeated_position_is_not_independent_evidence():
    p,s,r,v=fixture()
    for item in v:
        item['camera_center_m']=np.zeros(3);item['world_to_camera']=np.eye(4)
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'].any()


def test_fabricated_camera_separation_is_rejected():
    p,s,r,v=fixture()
    for item in v:item['world_to_camera']=np.eye(4)
    with pytest.raises(OnlineError,match='baselines'):
        review_depth_conflicts(p,s,r,v,clearance=.04)


def test_rotated_box_far_corner_is_protected():
    p,s,r,v=fixture();s[0]=[.2,.02,.2]
    a=2**-.5;r[0]=[[a,0,a],[0,1,0],[-a,0,a]]
    for item in v:item['depth'][:]=2.75
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'][0]


def test_full_shape_and_border_are_protected():
    p,s,r,v=fixture()
    s[0]=[.8,.8,.8]
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'].any()
    s[0]=.015;p[0,0]=1.25
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'][0]


def test_duplicate_captures_or_scaled_camera_rejected():
    p,s,r,v=fixture();v[1]['capture_id']=v[0]['capture_id']
    with pytest.raises(OnlineError):review_depth_conflicts(p,s,r,v,clearance=.04)
    v[1]['capture_id']='1';v[0]['world_to_camera'][0,0]=2
    with pytest.raises(OnlineError):review_depth_conflicts(p,s,r,v,clearance=.04)


def test_anisotropic_rotation_used_for_extent():
    p,s,r,v=fixture();s[0]=[.8,.01,.01]
    # Wide patch spills outside available image support and cannot be erased.
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'][0]
    # Turn its long axis along z: now it crosses the camera and cannot be erased.
    r[0]=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    assert not review_depth_conflicts(p,s,r,v,clearance=.04)['candidate_mask'][0]
