"""Evidence-supported room geometry and optional architectural layout diagnostics.

The layout functions generate an explicitly inferred room base.  The surface
constraint function retains the original scan topology and foreground geometry;
an inferred room base must not be substituted for a complete 3D reconstruction.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label


def constrain_room_surfaces(vertices, faces, points, normals, basis, bounds,
                            max_displacement=None, support_radius=None):
    """Straighten supported wall/floor interiors without replacing the scan.

    All existing faces are retained.  A vertex can move only when every incident
    triangle has the same supported architectural classification.  Angled faces,
    unsupported geometry, coherent raised surfaces, and their boundary collar
    are pinned to their exact input positions.  These conservative tests favor
    retaining uncertain 3D detail over flattening it into a nearby wall.

    ``basis`` maps raw coordinates to room x/y/up, and can be reflected.  Bounds
    supply the four wall offsets and floor offset; they do not describe hidden
    surfaces to generate.  Distances are reconstruction units, not meters.
    Returns new vertices, per-vertex plane labels (-1 means pinned), and a
    JSON-serializable geometric review.  Callers retain the input faces unchanged.
    """
    v = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError('Room surface triangle indices must be integers')
    f = raw_faces.astype(np.int64, copy=False)
    p = np.asarray(points, dtype=np.float64)
    pn = np.asarray(normals, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) == 0 or
        f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or
        p.ndim != 2 or p.shape[1] != 3 or len(p) < 8 or pn.shape != p.shape or
        basis.shape != (3, 3) or bounds.shape != (2, 3) or
        not all(np.isfinite(a).all() for a in (v, p, pn, basis, bounds)) or
        not np.allclose(basis.T@basis, np.eye(3), atol=1e-8) or
        np.any(bounds[1] <= bounds[0]) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('Invalid detail-preserving room surface inputs')
    diagonal = float(np.linalg.norm(bounds[1]-bounds[0]))
    maximum = diagonal*.006 if max_displacement is None else float(max_displacement)
    radius = diagonal*.004 if support_radius is None else float(support_radius)
    if (not np.isfinite(maximum) or not np.isfinite(radius) or maximum <= 0 or
        radius <= 0 or maximum > diagonal*.02 or radius > diagonal*.03):
        raise ValueError('Room surface constraint budget is outside conservative bounds')

    local = v@basis
    cloud_local = p@basis
    lengths = np.linalg.norm(pn, axis=1)
    pn = pn/np.maximum(lengths[:, None], 1e-12)
    cross = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    areas = np.linalg.norm(cross, axis=1)
    face_normals = cross/np.maximum(areas[:, None], 1e-12)
    tree = cKDTree(p)
    distance, nearest = tree.query(v, workers=2)
    point_angle = np.cos(np.deg2rad(25.))
    face_angle = np.cos(np.deg2rad(32.))
    face_labels = np.full(len(f), -1, dtype=np.int8)
    face_scores = np.full(len(f), np.inf)
    coherent_detail = np.zeros(len(v), dtype=bool)
    coherent_offset_detail = np.zeros(len(v), dtype=bool)
    coherent_tilted_detail = np.zeros(len(v), dtype=bool)
    surfaces = []
    plane_specs = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))
    for plane_id, (axis, side) in enumerate(plane_specs):
        normal = basis[:, axis]
        residual = local[:, axis]-bounds[side, axis]
        free = [i for i in range(3) if i != axis]
        supported = ((distance <= radius) & (np.abs(residual) <= maximum) &
                     (np.abs(pn[nearest]@normal) >= point_angle) &
                     (lengths[nearest] > 1e-12) &
                     (local[:, free] >= bounds[0, free]-radius).all(axis=1) &
                     (local[:, free] <= bounds[1, free]+radius).all(axis=1))

        # A flat, parallel foreground surface can pass a normal test.  Local
        # observed depth distinguishes a stable raised patch from scatter about
        # the structural plane.  Uncertain coherent offsets stay three-dimensional.
        detail = np.zeros(len(v), dtype=bool)
        offset_detail = np.zeros(len(v), dtype=bool)
        candidates = np.flatnonzero(supported)
        for start in range(0, len(candidates), 32768):
            index = candidates[start:start+32768]
            neighbor_distances, neighbors = tree.query(v[index], k=16,
                    distance_upper_bound=radius*2.5, workers=2)
            valid = (neighbors < len(p)) & np.isfinite(neighbor_distances)
            safe = np.minimum(neighbors, len(p)-1)
            valid &= ((np.abs(pn[safe]@normal) >= point_angle) &
                      (lengths[safe] > 1e-12))
            # A nearly closed door can be within the displacement slab and
            # inside the broad wall normal gate.  A consistently tilted local
            # observed surface is detail, rather than random wall scatter.
            observed = pn[safe]
            observed = observed*np.where(observed@normal >= 0., 1., -1.)[:, :, None]
            closest = pn[nearest[index]]
            closest = closest*np.where(closest@normal >= 0., 1., -1.)[:, None]
            agreement = np.sum(observed*closest[:, None], axis=2)
            local_count = valid.sum(axis=1)
            stable = (np.sum(np.where(valid, agreement, 0.), axis=1)/
                      np.maximum(local_count, 1)) >= np.cos(np.deg2rad(12.))
            tilted = ((closest@normal < np.cos(np.deg2rad(8.))) & stable &
                      (local_count >= 8))
            detail[index[tilted]] = True
            coherent_tilted_detail[index[tilted]] = True
            depths = cloud_local[safe, axis]-bounds[side, axis]
            valid &= np.abs(depths) <= maximum*1.25
            # Each row needs a genuine local patch, not one isolated point.
            count = valid.sum(axis=1)
            depths = np.where(valid, depths, np.nan)
            use = count >= 8
            if not use.any():
                continue
            medians = np.nanmedian(depths[use], axis=1)
            scatter = np.nanquantile(np.abs(depths[use]-medians[:, None]), .8, axis=1)
            raised = ((np.abs(medians) >= maximum*.3) & (scatter <= maximum*.1) &
                      (np.abs(residual[index[use]]-medians) <= maximum*.15))
            detail[index[use][raised]] = True
            offset_detail[index[use][raised]] = True
        coherent_detail |= detail
        coherent_offset_detail |= offset_detail

        normal_ok = np.abs(face_normals@normal) >= face_angle
        eligible_faces = (supported[f].all(axis=1) & normal_ok &
                          ~detail[f].any(axis=1) & (areas > 1e-12))
        score = np.max(np.abs(residual[f]), axis=1)
        choose = eligible_faces & (score < face_scores)
        face_labels[choose] = plane_id
        face_scores[choose] = score[choose]
        surfaces.append({'plane_id': plane_id, 'axis': axis, 'side': side,
            'offset': float(bounds[side, axis]), 'normal': normal.tolist(),
            'supported_vertices': int(supported.sum()),
            'candidate_faces': int(eligible_faces.sum()),
            'coherent_offset_detail_vertices': int(offset_detail.sum()),
            'coherent_detail_vertices': int(detail.sum())})

    # Protect complete object/unsupported triangles, including wall-shared hinge
    # vertices.  A normal averaged across a hinge must never authorize its move.
    face_labels[coherent_detail[f].any(axis=1)] = -1
    low = np.full(len(v), 5, dtype=np.int8)
    high = np.full(len(v), -1, dtype=np.int8)
    for corner in range(3):
        np.minimum.at(low, f[:, corner], face_labels)
        np.maximum.at(high, f[:, corner], face_labels)
    protected = (low < 0) | (low != high) | (high < 0)
    edges = np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    edge_vertices, incidence = np.unique(edges, axis=0, return_counts=True)
    boundary_vertices = np.unique(edge_vertices[incidence != 2])
    protected[boundary_vertices] = True
    initial_protected = protected.copy()
    # One further edge-connected collar isolates moved wall interiors from
    # unchanged door silhouettes, raised object rims, and hole boundaries.
    collar_faces = protected[f].any(axis=1)
    protected[np.unique(f[collar_faces])] = True
    labels = np.where(protected, -1, low).astype(np.int8)
    out = v.copy()
    for plane_id, (axis, side) in enumerate(plane_specs):
        selected = labels == plane_id
        residual = local[selected, axis]-bounds[side, axis]
        # Ease the displacement cap; do not stretch a barely eligible face by
        # making it jump the full distance onto the architectural plane.
        blend = np.clip((maximum-np.abs(residual))/(maximum*.25), 0., 1.)
        out[selected] -= (residual*blend)[:, None]*basis[:, axis]

    reverted = np.zeros(len(v), dtype=bool)
    original_valid = areas > 1e-12
    for _ in range(24):
        new_cross = np.cross(out[f[:, 1]]-out[f[:, 0]], out[f[:, 2]]-out[f[:, 0]])
        new_areas = np.linalg.norm(new_cross, axis=1)
        bad = original_valid & ((new_areas < areas*.1) |
                                (np.sum(cross*new_cross, axis=1) <= 0))
        if not bad.any():
            break
        undo = np.unique(f[bad])
        undo = undo[labels[undo] >= 0]
        if len(undo) == 0:
            raise ValueError('Room constraint could not preserve original triangle orientation')
        out[undo] = v[undo]
        labels[undo] = -1
        reverted[undo] = True
    else:
        raise ValueError('Room constraint topology guard did not converge')

    movement = np.linalg.norm(out-v, axis=1)
    if movement.max(initial=0) > maximum*(1+1e-8):
        raise ValueError('Room constraint exceeded its displacement bound')
    if not np.array_equal(out[labels < 0], v[labels < 0]):
        raise ValueError('A protected object or unsupported vertex was modified')
    for surface in surfaces:
        selected = labels == surface['plane_id']
        axis, side = surface['axis'], surface['side']
        surface['constrained_vertices'] = int(selected.sum())
        surface['moved_vertices'] = int((selected & (movement > 1e-8)).sum())
        if selected.any():
            before = local[selected, axis]-bounds[side, axis]
            after = out[selected]@basis[:, axis]-bounds[side, axis]
            surface['mesh_rms_before'] = float(np.sqrt(np.mean(before**2)))
            surface['mesh_rms_after'] = float(np.sqrt(np.mean(after**2)))
    return out, labels, {'method': 'detail_preserving_observed_room_surfaces',
        'surfaces': surfaces, 'moved_vertices': int((movement > 1e-8).sum()),
        'protected_vertices': int(initial_protected.sum()),
        'boundary_collar_vertices': int((protected & ~initial_protected).sum()),
        'open_or_nonmanifold_boundary_vertices': int(len(boundary_vertices)),
        'coherent_detail_vertices': int(coherent_detail.sum()),
        'coherent_offset_detail_vertices': int(coherent_offset_detail.sum()),
        'coherent_tilted_detail_vertices': int(coherent_tilted_detail.sum()),
        'unsupported_vertices': int((distance > radius).sum()),
        'object_or_unsupported_faces': int((face_labels < 0).sum()),
        'reverted_vertices': int(reverted.sum()),
        'max_displacement': float(movement.max(initial=0)),
        'input_vertices': len(v), 'output_vertices': len(out),
        'input_faces': len(f), 'output_faces': len(f), 'faces_added': 0, 'faces_removed': 0,
        'limits': {'displacement': maximum, 'support_radius': radius,
            'point_normal_angle_degrees': 25, 'face_normal_angle_degrees': 32,
            'coherent_offset_minimum': maximum*.3, 'coherent_offset_scatter': maximum*.1,
            'coherent_tilt_minimum_degrees': 8, 'coherent_normal_agreement_degrees': 12,
            'minimum_local_support_points': 8, 'boundary_collar_hops': 1,
            'minimum_triangle_area_ratio': .1},
        'units': 'COLMAP reconstruction units; metric scale unverified',
        'preserved_detail': 'Angled, raised, unsupported and architectural-boundary triangles keep exact source positions',
        'inferred_surfaces': 'none; original holes and topology retained',
        'limitations': 'Conservative geometric classification is not semantic recognition; coherent uncertain wall offsets are retained and missing object geometry cannot be invented'}


def select_room_planes(points, normals, patches, axes, camera_centers, vertical_axis, up_sign):
    if (points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or
        len(points) < 1000 or camera_centers.ndim != 2 or camera_centers.shape[1] != 3 or
        len(camera_centers) < 3 or not all(np.isfinite(a).all() for a in (points, normals, axes, camera_centers)) or
        axes.shape != (3,3) or not np.allclose(axes.T@axes, np.eye(3), atol=1e-8) or
        vertical_axis not in (0,1,2) or up_sign not in (-1,1)):
        raise ValueError('Invalid observed room geometry or camera frame')
    horizontal = [i for i in range(3) if i != vertical_axis]
    basis = np.column_stack([axes[:, i] for i in horizontal]+[axes[:, vertical_axis]*up_sign])
    cloud, cameras = points@basis, camera_centers@basis
    diag = float(np.linalg.norm(np.quantile(cloud,.99,axis=0)-np.quantile(cloud,.01,axis=0)))
    bounds = np.empty((2,3)); evidence=[]
    for direction, raw_axis in enumerate(horizontal):
        candidates = []
        for patch in patches:
            if patch['room_axis'] == raw_axis:
                candidates.append((float(np.array(patch['origin'])@basis[:,direction]),
                                   float(np.prod(patch['bounds']['extent'][:2]))))
        candidates.sort()
        groups=[]
        for offset, area in candidates:
            if groups and abs(offset-np.average(groups[-1][0],weights=groups[-1][1])) <= .015*diag:
                groups[-1][0].append(offset); groups[-1][1].append(area)
            else: groups.append(([offset],[area]))
        groups=[(float(np.average(offsets,weights=weights)),sum(weights)) for offsets,weights in groups]
        low,high=np.quantile(cameras[:,direction],[.2,.8])
        left=[g for g in groups if g[0]<low]; right=[g for g in groups if g[0]>high]
        if not left or not right: raise ValueError('Both opposing structural walls need observed plane support')
        chosen=[max(left,key=lambda g:g[1]),max(right,key=lambda g:g[1])]
        for side,(offset,area) in enumerate(chosen):
            support=(np.abs(cloud[:,direction]-offset)<.015*diag)&(np.abs(normals@basis[:,direction])>np.cos(np.deg2rad(25)))
            if support.sum()<500: raise ValueError('Insufficient observed wall support')
            bounds[side,direction]=float(np.median(cloud[support,direction]))
            evidence.append({'axis':direction,'side':side,'offset':bounds[side,direction],
                             'observed_points':int(support.sum()),'patch_area':area})
    floor_patches=[p for p in patches if p['room_axis']==vertical_axis]
    if not floor_patches: raise ValueError('A supported room floor is required')
    largest=max(float(np.prod(p['bounds']['extent'][:2])) for p in floor_patches)
    floor_patches=[p for p in floor_patches if np.prod(p['bounds']['extent'][:2]) >= largest*.5]
    floor=min(float(np.array(p['origin'])@basis[:,2]) for p in floor_patches)
    support=(np.abs(cloud[:,2]-floor)<.015*diag)&(np.abs(normals@basis[:,2])>np.cos(np.deg2rad(25)))
    if support.sum()<500: raise ValueError('Insufficient observed floor support')
    bounds[0,2]=float(np.median(cloud[support,2]))
    interior=(cloud[:,:2]>=bounds[0,:2]).all(1)&(cloud[:,:2]<=bounds[1,:2]).all(1)
    bounds[1,2]=float(np.quantile(cloud[interior,2],.995))
    if np.any(bounds[1]-bounds[0]<=0): raise ValueError('Invalid room extent')
    return basis,bounds,diag,evidence


def plane_grids(bounds, spacing):
    """Five exact planar surfaces; ceiling is not inferred from an unseen capture."""
    if (np.shape(bounds) != (2,3) or not np.isfinite(bounds).all() or
        not np.isfinite(spacing) or spacing <= 0 or np.any(bounds[1] <= bounds[0])):
        raise ValueError('Invalid room bounds or surface spacing')
    result=[]
    for fixed,side in ((0,0),(0,1),(1,0),(1,1),(2,0)):
        free=[i for i in range(3) if i!=fixed]
        count=np.ceil((bounds[1,free]-bounds[0,free])/spacing).astype(int)
        if np.prod(count)>150000: raise ValueError('Architectural surface budget exceeded')
        values=[np.linspace(bounds[0,i],bounds[1,i],int(n)+1) for i,n in zip(free,count)]
        u,v=np.meshgrid(*values,indexing='xy')
        vertices=np.zeros((u.size,3)); vertices[:,fixed]=bounds[side,fixed]
        vertices[:,free[0]]=u.ravel(); vertices[:,free[1]]=v.ravel()
        x,y=np.meshgrid(np.arange(count[0]),np.arange(count[1]),indexing='xy')
        corner=(y*(count[0]+1)+x).ravel()
        faces=np.stack((np.column_stack((corner,corner+1,corner+count[0]+1)),
                        np.column_stack((corner+1,corner+count[0]+2,corner+count[0]+1))),axis=1)
        desired=np.zeros(3);desired[fixed]=1 if side==0 else -1
        triangle=vertices[faces[0,0]]
        if np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])@desired<0: faces=faces[:,:,::-1]
        result.append({'fixed':fixed,'side':side,'free':free,'count':count,'vertices':vertices,
                       'faces':faces,'votes':np.zeros(int(np.prod(count)),dtype=np.uint16)})
    return result


def add_opening_evidence(grids,bounds,origin,directions,depths,tolerance):
    """Carve only rays with consistent measured depth beyond a structural wall."""
    if directions.shape != (len(depths),3) or tolerance <= 0 or not np.isfinite(origin).all():
        raise ValueError('Invalid calibrated opening rays')
    distances=[]
    with np.errstate(divide='ignore',invalid='ignore'):
        for grid in grids:
            axis=grid['fixed']; t=(bounds[grid['side'],axis]-origin[axis])/directions[:,axis]
            hits=origin+directions*t[:,None]
            free=grid['free']
            valid=(t>0)&np.isfinite(t)&(hits[:,free]>=bounds[0,free]-.001).all(1)&(hits[:,free]<=bounds[1,free]+.001).all(1)
            distances.append(np.where(valid,t,np.inf))
    distances=np.array(distances)
    first=np.argmin(distances,axis=0); nearest=distances[first,np.arange(len(depths))]
    beyond=np.isfinite(nearest)&np.isfinite(depths)&(depths>nearest+tolerance)&(depths>0)
    for index,grid in enumerate(grids[:4]):
        use=beyond&(first==index)
        hit=origin+directions[use]*nearest[use,None]
        free=grid['free']; count=grid['count']
        cell=np.floor((hit[:,free]-bounds[0,free])/(bounds[1,free]-bounds[0,free])*count).astype(int)
        cell=np.clip(cell,0,count-1)
        cells=np.unique(cell[:,1]*count[0]+cell[:,0])
        grid['votes'][cells]+=1  # Each camera contributes at most one vote per cell.


def build_room_surface(points,colors,basis,bounds,grids,minimum_votes=3):
    """Construct the declared structural planes, retaining depth-proven openings."""
    if minimum_votes < 3 or colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError('Opening evidence and photo colors are required')
    vertices=[];faces=[];offset=0;reports=[]
    for grid in grids:
        count=grid['count']; holes=(grid['votes']>=minimum_votes).reshape(count[1],count[0])
        components,number=label(holes)
        accepted=np.zeros_like(holes)
        for component in range(1,number+1):
            ys,xs=np.nonzero(components==component)
            if len(xs)<max(8,int(holes.size*.001)): continue
            # Straight opening edges are a declared architectural prior; keep
            # sparse scattered rays from carving a spurious window or doorway.
            occupancy=len(xs)/((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))
            if occupancy>=.3: accepted[ys.min():ys.max()+1,xs.min():xs.max()+1]=True
            else: accepted[components==component]=True
        selected=grid['faces'][~accepted.ravel()].reshape(-1,3)
        vertices.append(grid['vertices']);faces.append(selected+offset);offset+=len(grid['vertices'])
        reports.append({'axis':grid['fixed'],'side':grid['side'],'cells':int(holes.size),
                        'depth_proven_opening_cells':int(accepted.sum()),'maximum_camera_votes':int(grid['votes'].max(initial=0))})
    v=np.concatenate(vertices)@basis.T; f=np.concatenate(faces)
    # A reflected room basis must preserve inward-facing triangle orientation.
    if np.linalg.det(basis)<0:f=f[:,::-1]
    distance,index=cKDTree(points).query(v,workers=2)
    return v,f,colors[index],{'surfaces':reports,'room_basis':basis.tolist(),'room_bounds':bounds.tolist(),
        'spans_reconstruction_units':(bounds[1]-bounds[0]).tolist(),'corner_angles_degrees':[90,90,90],
        'structural_prior':'flat floor and four straight perpendicular/parallel walls',
        'ceiling':'not generated; upper wall bound follows observed extent',
        'physical_dimensions':'unmeasured; equal wall lengths not forced',
        'foreground_objects':'retained in the separate detailed scan, not reconstructed as architectural planes',
        'color_interpolation_distance_p90':float(np.quantile(distance,.9)),
        'inferred_surfaces':'structural planes continue behind occluders; these are not measured hidden surfaces'}
