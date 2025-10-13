import igl
import numpy as np
import torch

from tools.param import cross_epsilon


def flood_ext(ext_fid, boundary_edges, f):
    def adjacent_edge(fid1, fid2):
        e_set = []
        for i in range(3):
            e = (min(f[fid1][i], f[fid1][(i+1) % 3]),
                 max(f[fid1][i], f[fid1][(i+1) % 3]))
            e_set.append(e)
        for i in range(3):
            e = (min(f[fid2][i], f[fid2][(i+1) % 3]),
                 max(f[fid2][i], f[fid2][(i+1) % 3]))
            if e in e_set:
                return e

        return None
    # Flood from the exterior seed triangles without crossing boundary edges
    tt, _ = igl.triangle_triangle_adjacency(f)
    f_k_ring = ext_fid
    for _ in range(f.shape[0]):
        new_f = tt[f_k_ring].ravel().tolist()
        filtered_new_f = []
        for fid in new_f:
            if fid < 0 or fid in f_k_ring:
                continue

            cross_boundary = False
            # Get the traversed edge
            for fid2 in tt[fid]:
                if fid2 >= 0 and fid2 in f_k_ring:
                    e = adjacent_edge(fid, fid2)
                    if e in boundary_edges:
                        cross_boundary = True
                        break
            if not cross_boundary:
                filtered_new_f.append(fid)
        if len(filtered_new_f) > 0:
            f_k_ring = list(set(f_k_ring + filtered_new_f))
        else:
            break

    return f_k_ring


def poly_edge_correspondence(polys, mesh, threshold=1):
    def edges_dist(poly, points_in):
        # Get the edge midpoint
        points = points_in[:, 0:2]

        poly0 = np.array(poly[0:-1])
        poly1 = np.array(poly[1::])

        line_vector = poly1 - poly0
        line_vector = line_vector[np.newaxis, :, :]
        point_vector = points[:, np.newaxis, :] - poly0
        line_length = np.sum(line_vector * line_vector, axis=2)
        line_length[line_length == 0] = 1
        projection = np.sum(point_vector * line_vector, axis=2) / line_length
        projection = np.clip(projection, 0, 1)
        closest_point = poly0[np.newaxis, :, :] + \
            projection[..., np.newaxis] * line_vector

        # Calculating the distance
        distances = np.sqrt(
            np.sum((points[:, np.newaxis, :] - closest_point) ** 2, axis=-1))

        return distances

    e_discont = []
    e_discont_ends = []
    e_corresp = []

    # Build edges from faces
    e = np.concatenate([mesh.f[:, [0, 1]], mesh.f[:, [1, 2]],
                        mesh.f[:, [2, 0]]], axis=0)
    vp_map = {}
    for pid, poly in enumerate(polys):
        points = (mesh.v[e[:, 0]] + mesh.v[e[:, 1]]) / 2
        # |E| x |Poly|
        distances = edges_dist(poly, points)
        indices = np.argmin(distances, axis=1)
        rows, cols = np.where(distances < threshold)
        ep_corresp = [(row, col) for row, col in zip(
            rows, cols) if col == indices[row]]
        for eid, i in ep_corresp:
            v0, v1 = e[eid]
            vp_map[v0] = pid
            vp_map[v1] = pid
            e_discont.append((min(v0, v1), max(v0, v1)))

    endsp_map = {}
    for pid, poly in enumerate(polys):
        points = mesh.v[e[:, 0]]
        # |E| x |Poly|
        distances = edges_dist(poly, points)
        indices = np.argmin(distances, axis=1)
        rows, cols = np.where(distances < threshold)
        ep_corresp = [(row, col) for row, col in zip(
            rows, cols) if col == indices[row]]
        for eid, i in ep_corresp:
            v0, v1 = e[eid]
            endsp_map[v0] = pid

    for pid, poly in enumerate(polys):
        points = mesh.v[e[:, 1]]
        # |E| x |Poly|
        distances = edges_dist(poly, points)
        indices = np.argmin(distances, axis=1)
        rows, cols = np.where(distances < threshold)
        ep_corresp = [(row, col) for row, col in zip(
            rows, cols) if col == indices[row]]
        for eid, i in ep_corresp:
            v0, v1 = e[eid]
            if v0 in endsp_map:
                vp_map[v0] = pid
                vp_map[v1] = pid
                e_discont_ends.append((min(v0, v1), max(v0, v1)))

    # Intersect the saved edge sets
    e_discont = list(set(e_discont).intersection(set(e_discont_ends)))

    for vid in sorted(vp_map.keys()):
        # e_discont.append(vid)
        e_corresp.append(vp_map[vid])
    e_discont = np.array(e_discont)

    return e_discont, e_corresp


def compute_angles(dir_ref, dir_thetas, set_ref=False):
    dot_product = torch.sum(dir_ref * dir_thetas, dim=2)
    dir_thetas_norm = torch.linalg.norm(dir_thetas, dim=2)
    norm_product = torch.linalg.norm(dir_ref, dim=2) * dir_thetas_norm
    dot_nonzero = torch.where(
        dir_thetas_norm > 0, dot_product / norm_product, 0.0)
    cos_angle = torch.clamp(dot_nonzero, -1.0, 1.0)
    angle_init = torch.acos(cos_angle)
    cross_product = torch.linalg.cross(
        dir_ref, dir_thetas, dim=2) / norm_product.unsqueeze(-1)
    angle_ccw = torch.where(cross_product[:, :, 2] > cross_epsilon,
                            2 * torch.pi - angle_init, angle_init)
    angle_ccw = torch.fmod(angle_ccw + 2 * torch.pi, 2 * torch.pi)

    # Hard set the reference angle to be 0
    if set_ref:
        angle_ccw[:, 0] = 0

    return angle_ccw


def compute_angles_atan2(dir_ref, dir_thetas, set_ref=False):
    # % signed angle betxeen u and x
    #     theta = atan2(u(1)*x(2)-u(2)*x(1),u(1)*x(1)+u(2)*x(2));
    #     % normalize to [0,1]
    #     t = (theta+pi)/(2*pi);
    dir_thetas = -dir_thetas
    theta = torch.atan2(dir_thetas[..., 0] * dir_ref[..., 1]-dir_thetas[..., 1] * dir_ref[..., 0],
                        dir_thetas[..., 0] * dir_ref[..., 0]+dir_thetas[..., 1] * dir_ref[..., 1])
    angle_ccw = (theta + np.pi)
    angle_ccw = torch.fmod(angle_ccw, 2 * torch.pi)

    # Hard set the reference angle to be 0
    if set_ref:
        angle_ccw[:, 0] = 0

    return angle_ccw


def incircle(a, b, c, d):
    # TODO: Use actual predicates...
    # For now just make sure we are using the most precision
    a = a.astype(np.longdouble)
    b = b.astype(np.longdouble)
    c = c.astype(np.longdouble)
    d = d.astype(np.longdouble)

    det = np.array([[a[0] - d[0], a[1] - d[1], (a[0] - d[0])**2 + (a[1] - d[1])**2],
                    [b[0] - d[0], b[1] - d[1],
                        (b[0] - d[0])**2 + (b[1] - d[1])**2],
                    [c[0] - d[0], c[1] - d[1], (c[0] - d[0])**2 + (c[1] - d[1])**2]])
    # is_inside = np.linalg.det(det)
    # Manually compute the 3x3 determinant since the type is not supported
    is_inside = det[0, 0] * det[1, 1] * det[2, 2] + det[0, 1] * det[1, 2] * det[2, 0] + \
        det[0, 2] * det[1, 0] * det[2, 1] - det[0, 2] * det[1, 1] * det[2, 0] - \
        det[0, 1] * det[1, 0] * det[2, 2] - det[0, 0] * det[1, 2] * det[2, 1]

    return is_inside
