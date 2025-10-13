import time

from gpytoolbox import ray_mesh_intersect
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_scatter import scatter_sum

from geometry.defgrid import find_k_ring_vf
from geometry.softras import is_point_in_triangle_mat
from neural.utils import barycentric, render_face
from tools.plot_utils import plot_association_func, plot_mesh, to_pil_image
from tools.param import epsilon


def point_triangle_squared_distance_closest_mat(samples, p_indices_extended, v, f):
    def point_edge_distance(points, v0, v1, center):
        line_vector = (v1 - v0)
        point_vector = points - v0
        projection = torch.sum(point_vector * line_vector, dim=-1) / \
            torch.sum(line_vector * line_vector, dim=-1)
        projection = torch.clamp(projection, 0, 1)
        closest_point = v0 + projection.unsqueeze(-1) * line_vector

        # Soft rasterizer paper uses d^2
        distance = torch.sum((points - closest_point) ** 2, dim=-1)

        # Move slightly toward the center to avoid degenerate case of hitting the edges
        center_dir = (center - closest_point)
        center_dir = center_dir / torch.norm(center_dir, dim=-1, keepdim=True)
        closest_point = closest_point + center_dir * 1e-6

        return distance, closest_point

    V0 = v[f[:, 0]].unsqueeze(0)
    V1 = v[f[:, 1]].unsqueeze(0)
    V2 = v[f[:, 2]].unsqueeze(0)

    samples_picked = samples[p_indices_extended]
    samples_picked = samples[p_indices_extended]

    center = (V0 + V1 + V2) / 3

    d01, c01 = point_edge_distance(samples_picked, V0, V1, center)
    d12, c12 = point_edge_distance(samples_picked, V1, V2, center)
    d20, c20 = point_edge_distance(samples_picked, V2, V0, center)

    distances = torch.stack([d01, d12, d20])
    min_indices = torch.min(distances, dim=0).indices
    min_indices = min_indices.unsqueeze(-1)
    # closest_points_all = torch.stack([c01, c12, c20])
    closest_points = torch.zeros_like(c01)
    closest_points = torch.where(min_indices == 0, c01, closest_points)
    closest_points = torch.where(min_indices == 1, c12, closest_points)
    closest_points = torch.where(min_indices == 2, c20, closest_points)
    distances = torch.min(distances, dim=0).values

    return distances.to(v.dtype), closest_points.to(v.dtype)


def softras_render(mlp, x, sigma, k_ring=1, to_normalize=True, return_spp=False):
    mesh = mlp.mesh

    # Look up features for interpolation
    # Reference: x contains [float(y) / height, float(x) / width]
    c = np.column_stack(
        (x.detach().cpu().numpy(), -1 * np.ones(x.shape[0])))
    c[:, [0, 1]] = c[:, [1, 0]]
    c[:, 0] *= mlp.mesh.size[0]
    c[:, 1] *= mlp.mesh.size[1]

    # # Offset the center
    # c += np.array([0.5, 0.5, 0])

    d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
    v = mlp.get_v()
    v_np = v.cpu().detach().numpy()
    v_np[:, 2] = 0
    _, ids, ll = ray_mesh_intersect(
        c, d, v_np, mlp.mesh.f.cpu().detach().numpy())

    # Differentiable computation of barycentric
    points = x[:, [1, 0]]
    points[:, 0] = points[:, 0] * mlp.mesh.size[0]
    points[:, 1] = points[:, 1] * mlp.mesh.size[1]
    l = barycentric(points, v[mlp.mesh.f[ids]])
    cc = torch.hstack([points, torch.zeros(
        [points.shape[0], 1], device=points.device)])

    # Compute soft occupancy using SoftRas
    # 1. Associate each sample with a neighborhood
    # |S| x 3
    with torch.no_grad():
        sample_start_time = time.time()

        samples = c
        in_faces = ids
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).to(mesh.device)
        if isinstance(in_faces, np.ndarray):
            in_faces = torch.from_numpy(in_faces).to(mesh.device)

        faces = list(range(mesh.f.shape[0]))

        # Fetch the neighborhood per triangle per sample
        # Use the largest neighborhood size to make a dense but efficient matrix
        max_neighborhood_size = 0
        max_sample_size = 0
        p_indices = []
        t_indices = []

        k_ring_time = 0
        sample_lookup_time = 0
        save_time = 0
        kring_start_time = time.time()
        for fid in faces:
            k_f_start_time = time.time()

            # Use the k-ring as our domain
            f_k_ring = find_k_ring_vf(mesh.vf, mesh.f, fid, k_ring)

            f_k_ring = torch.tensor(f_k_ring)
            f_k_ring = f_k_ring.unsqueeze(1)
            max_neighborhood_size = max(
                max_neighborhood_size, f_k_ring.shape[0])
            f_k_ring = f_k_ring.to(mesh.device)

            k_f_end_time = time.time()
            k_f_time = k_f_end_time - k_f_start_time
            k_ring_time += k_f_time

            if f_k_ring.shape[0] == 0:
                continue

            int_f_k_ring = f_k_ring[f_k_ring < mesh.f_interior_count]
            if int_f_k_ring.shape[0] == 0:
                continue

            sample_lookup_start_time = time.time()
            # Find the samples within this k-ring neighborhood
            mask = torch.any(torch.isin(
                in_faces, int_f_k_ring).reshape(-1, 1), axis=1)
            sample_indices = torch.where(mask)[0]
            max_sample_size = max(max_sample_size, sample_indices.shape[0])

            sample_lookup_end_time = time.time()
            sample_lookup_time += sample_lookup_end_time - sample_lookup_start_time

            p_indices.append(sample_indices.int().reshape(-1, 1))
            t_indices.append(int(fid))

            save_time += time.time() - sample_lookup_end_time

        kring_end_time = time.time()
        kring_execution_time = kring_end_time - kring_start_time

        p_indices = torch.nn.utils.rnn.pad_sequence(
            p_indices, batch_first=True, padding_value=-1)
        # Smax x |F|
        p_indices = p_indices.squeeze(-1).T
        p_indices_extended = torch.clamp(p_indices, min=0)

    # Compute inside/outside and triangle distance for each sample-triangle pair
    # D_ij = sigmoid(δ_ij · d^2(i, j)/σ)
    # Smax x |F|
    inside_triangle = is_point_in_triangle_mat(
        samples, p_indices_extended, v, mesh.f[t_indices])
    squared_distances, closest_points = point_triangle_squared_distance_closest_mat(
        samples, p_indices_extended, v, mesh.f[t_indices])
    sign_squared_distances = (2 * inside_triangle - 1) * squared_distances
    p_dist_raw = sign_squared_distances / sigma
    p_sigmoid_raw = torch.sigmoid(p_dist_raw)

    # Render closest points for the outside samples
    with torch.no_grad():
        column_indices = torch.arange(inside_triangle.shape[1])
        expanded_indices = column_indices.unsqueeze(
            0).expand(inside_triangle.shape[0], -1)
        expanded_indices = expanded_indices.to(inside_triangle.device)
        outside_faces = expanded_indices[torch.logical_and(
            p_indices >= 0, ~(inside_triangle.bool()))]
        outside_query = closest_points[torch.logical_and(
            p_indices >= 0, ~(inside_triangle.bool()))]

        # closest_l = barycentric(samples[p_indices_extended[torch.logical_and(
        #     p_indices >= 0, ~(inside_triangle.bool()))]], v[mesh.f[outside_faces]])
        # closest_l = torch.clamp(closest_l, epsilon, 1 - epsilon)
        # closest_l[:, 2] = 1 - closest_l[:, 0] - closest_l[:, 1]
        # outside_query = closest_l[:, 0].unsqueeze(-1) * v[mesh.f[outside_faces, 0]] + \
        #     closest_l[:, 1].unsqueeze(-1) * v[mesh.f[outside_faces, 1]] + \
        #     closest_l[:, 2].unsqueeze(-1) * v[mesh.f[outside_faces, 2]]

        closest_colors = render_face(mlp, outside_faces, outside_query)
    closest_colors_dummy = torch.concat((torch.full(
        (1, closest_colors.shape[-1]), fill_value=0, device=closest_colors.device), closest_colors), dim=0)

    # Compute inside colors
    colors_trimmed, target_rows = mlp.interpolate(ids, l, c, cc)
    colors = torch.zeros(
        (x.shape[0], colors_trimmed.shape[1]), device=colors_trimmed.device)
    colors[target_rows] = colors[target_rows] + colors_trimmed

    # Add dummy
    colors_dummy = torch.concat((torch.full(
        (1, colors.shape[-1]), fill_value=0, device=colors.device), colors), dim=0)
    p_indices_dummy = p_indices + 1
    colors_assembled = torch.zeros(
        (p_indices.shape[0], p_indices.shape[1], colors.shape[-1]), device=colors.device)
    colors_assembled = torch.where(
        inside_triangle.bool().unsqueeze(-1), colors_dummy[p_indices_dummy], 0)
    # colors_assembled = torch.where(torch.logical_and(p_indices >= 0, ~(
    #     inside_triangle.bool())).unsqueeze(-1), closest_colors, colors_assembled)
    if not colors_assembled.is_contiguous():
        colors_assembled = colors_assembled.contiguous()
    colors_assembled_flatten = colors_assembled.view(-1, colors.shape[-1])
    with torch.no_grad():
        color_indices = -1 * torch.ones_like(p_indices)
        closest_indices = torch.arange(
            closest_colors.shape[0], device=closest_colors.device)
        color_indices = color_indices.to(closest_indices.dtype)
        color_indices[torch.logical_and(
            p_indices >= 0, ~(inside_triangle.bool()))] = closest_indices
        color_indices = color_indices + 1
        color_indices = color_indices.ravel()
    colors_assembled_flatten[..., 0] = colors_assembled_flatten[...,
                                                                0] + closest_colors_dummy[..., 0].ravel()[color_indices]
    colors_assembled_flatten[..., 1] = colors_assembled_flatten[...,
                                                                1] + closest_colors_dummy[..., 1].ravel()[color_indices]
    colors_assembled_flatten[..., 2] = colors_assembled_flatten[...,
                                                                2] + closest_colors_dummy[..., 2].ravel()[color_indices]

    # Set the values corresponding to invalid neighor triangles to 0
    p_sigmoid = torch.where(p_indices < 0, torch.tensor(0.), p_sigmoid_raw)

    # if to_normalize:
    #     p_count = torch.where(p_indices < 0, torch.tensor(0.),
    #                           torch.ones_like(p_sigmoid))
    # else:
    #     p_count = torch.zeros_like(p_sigmoid)

    color_soft = (p_sigmoid.unsqueeze(-1) * colors_assembled)
    c0 = torch.zeros(
        (colors_dummy.shape[0]), dtype=colors_dummy.dtype, device=colors_dummy.device)
    c1 = torch.zeros_like(c0)
    c2 = torch.zeros_like(c0)
    scatter_sum(
        src=color_soft[..., 0].ravel(), index=p_indices_dummy.long().ravel(), out=c0)
    scatter_sum(
        src=color_soft[..., 1].ravel(), index=p_indices_dummy.long().ravel(), out=c1)
    scatter_sum(
        src=color_soft[..., 2].ravel(), index=p_indices_dummy.long().ravel(), out=c2)
    color_soft_sum = torch.hstack(
        [c0.reshape(-1, 1), c1.reshape(-1, 1), c2.reshape(-1, 1)])

    if return_spp:
        int_counts = torch.ones_like(color_soft[..., 0]).to(mlp.mesh.f.dtype)
        canvas_spp = torch.zeros(
            (colors_dummy.shape[0]), dtype=mlp.mesh.f.dtype, device=colors_dummy.device)
        scatter_sum(
            src=int_counts.ravel(), index=p_indices_dummy.long().ravel(), out=canvas_spp)

        return color_soft_sum[1::, ...], canvas_spp[1::, ...]

    with torch.no_grad():
        if False:
            samples_pixel = torch.floor(samples).int()
            samples_pixel[:, 0] = torch.clamp(
                samples_pixel[:, 0], 0, mlp.image.width - 1)
            samples_pixel[:, 1] = torch.clamp(
                samples_pixel[:, 1], 0, mlp.image.height - 1)

            if not hasattr(mlp, 'canvas_debug'):
                mlp.canvas_debug = torch.zeros(
                    [mlp.image.height, mlp.image.width, canvas.shape[1]])
                mlp.canvas_debug = mlp.canvas_debug.to(mlp.device)

            canvas_debug2 = torch.zeros(
                [mlp.image.height, mlp.image.width, canvas.shape[1]])
            canvas_debug2 = canvas_debug2.to(mlp.device)
            canvas_debug2[samples_pixel[:, 1],
                          samples_pixel[:, 0], :] = canvas
            mlp.canvas_debug = mlp.canvas_debug + canvas_debug2

        if False:
            # if True:
            # if not hasattr(mlp, 'canvas_debug'):
            fid = 3155
            ff_debug = mesh.f[fid]
            print(v[ff_debug])

            color_debug = torch.zeros(
                samples.shape[0], 3, dtype=v.dtype, device=mesh.device)
            p_indices_face = p_indices[:, fid]
            color_face = color_soft[:, fid, ...]
            # color_face = torch.hstack([p_sigmoid.unsqueeze(-1)[:, fid, ...], p_sigmoid.unsqueeze(-1)[
            #                           :, fid, ...], p_sigmoid.unsqueeze(-1)[:, fid, ...]])
            # color_face = colors_assembled[:, fid, ...]
            p_indices_face_valid = p_indices_face[p_indices_face >= 0]
            color_debug[p_indices_face_valid] = color_face[p_indices_face >= 0]

            samples_pixel = torch.floor(samples).int()
            samples_pixel[:, 0] = torch.clamp(
                samples_pixel[:, 0], 0, mlp.image.width - 1)
            samples_pixel[:, 1] = torch.clamp(
                samples_pixel[:, 1], 0, mlp.image.height - 1)

            if not hasattr(mlp, 'canvas_debug'):
                mlp.canvas_debug = torch.zeros(
                    [mlp.image.height, mlp.image.width, color_debug.shape[1]])
                mlp.canvas_debug = mlp.canvas_debug.to(mlp.device)

            canvas_debug2 = torch.zeros(
                [mlp.image.height, mlp.image.width, color_debug.shape[1]])
            canvas_debug2 = canvas_debug2.to(mlp.device)
            canvas_debug2[samples_pixel[:, 1],
                          samples_pixel[:, 0], :] = color_debug
            mlp.canvas_debug = mlp.canvas_debug + canvas_debug2

    return color_soft_sum[1::, ...]
