import math
import time

import igl
from gpytoolbox import ray_mesh_intersect
import numpy as np
import torch
from torch_scatter import scatter_max


def stratify_1d(x):
    # Given x is evenly spaced in range [0, 1] using linspace
    # Stratify each sample in its own bin of size 1/(num_samples-1)
    num_samples = x.shape[0]
    bin_size = 1 / (num_samples - 1)
    sample_offset = torch.rand(num_samples, 1)
    sample_offset = sample_offset * bin_size - 0.5 * bin_size

    # Adjust the first the last sample to avoid out of range
    sample_offset[0] += 0.5 * bin_size
    sample_offset[-1] -= 0.5 * bin_size

    sample_offset[0] = 0.5 * sample_offset[0]
    sample_offset[-1] = 0.5 * sample_offset[-1]

    return x + sample_offset


def stratify_2d(x, bin_size=torch.tensor([[1, 1]])):
    assert x.shape[1] == 2, "Input must be 2D"

    # Stratify each sample in its own bin with a size given by bin_size
    num_samples = x.shape[0]
    sample_offset = torch.rand(num_samples, 2)
    sample_offset = sample_offset * bin_size - 0.5 * bin_size

    return x + sample_offset


def stratify_2d_offset(x, bin_size=torch.tensor([[1, 1]])):
    assert x.shape[1] == 2, "Input must be 2D"

    # Stratify each sample in its own bin with a size given by bin_size
    bin_size = bin_size.to(x.device)
    num_samples = x.shape[0]
    sample_offset = torch.rand(num_samples, 2, device=x.device)
    sample_offset = sample_offset * bin_size - 0.5 * bin_size

    return sample_offset


@torch.no_grad()
def prepare_interior_spp(mlp, spp, samples=None):
    image = mlp.image
    if not isinstance(samples, torch.Tensor):
        samples = subpixel_sample(image.width, image.height, spp)
        if samples.device != mlp.device:
            samples = samples.to(mlp.device)

    offset = stratify_2d_offset(samples, bin_size=torch.tensor(
        [[1.0 / spp, 1.0 / spp]]))
    offset[:, 0] = offset[:, 0] / image.height
    offset[:, 1] = offset[:, 1] / image.width
    samples = samples + offset

    return samples


@torch.no_grad()
def prepare_edge_spp(mlp, edge_spp, edge_epsilon=1e-4, importance_threshold=-1):
    num_edge_samples = edge_spp * mlp.image.size[0] * mlp.image.size[1]
    # debug-only override removed
    edge_samples_t, samples_ei, edge_normals, edge_length, e_prob = edge_sample(
        mlp, num_edge_samples, importance_threshold=importance_threshold)

    e = mlp.mesh_e
    if e.device != mlp.device:
        e = e.to(mlp.device)

    edge_samples = mlp.get_v()[e[samples_ei][:, 0]] + \
        edge_samples_t * \
        ((mlp.get_v()[e[samples_ei][:, 1]] -
         mlp.get_v()[e[samples_ei][:, 0]])/edge_length[samples_ei])
    edge_samples_left = edge_samples + edge_epsilon * edge_normals[samples_ei]
    edge_samples_right = edge_samples - edge_epsilon * edge_normals[samples_ei]

    # Filter the edge samples that falls into two bins
    same_bin = (torch.floor(edge_samples_left).long() ==
                torch.floor(edge_samples_right).long()).all(dim=1)

    edge_samples_left = edge_samples_left[same_bin]
    edge_samples_right = edge_samples_right[same_bin]
    edge_samples_left[:, 0] = edge_samples_left[:, 0] / mlp.image.size[0]
    edge_samples_left[:, 1] = edge_samples_left[:, 1] / mlp.image.size[1]
    edge_samples_right[:, 0] = edge_samples_right[:, 0] / mlp.image.size[0]
    edge_samples_right[:, 1] = edge_samples_right[:, 1] / mlp.image.size[1]

    edge_samples_t = edge_samples_t[same_bin]
    samples_ei = samples_ei[same_bin]

    edge_prob = e_prob[same_bin]

    return (edge_samples_left, edge_samples_right, samples_ei, edge_length,
            edge_normals, edge_samples_t, edge_prob)


def prepare_interior_data(spp, mlp, batch_size, int_samples_=None):
    # Prepare interior samples
    sqrt_spp = int(math.sqrt(spp))
    int_samples = prepare_interior_spp(mlp, sqrt_spp, samples=int_samples_)

    # Split data into batches
    int_samples_batches = int_samples.split(batch_size)

    return int_samples_batches


def prepare_edge_data(edge_spp, mlp, batch_size, edge_epsilon=1e-4, importance_threshold=-1):
    # Prepare edge samples
    edge_samples_ = prepare_edge_spp(
        mlp, edge_spp, edge_epsilon, importance_threshold=importance_threshold)

    # Split data into batches
    edge_samples_split = []
    for i, e_samples in enumerate(edge_samples_):
        if i == 3 or i == 4:
            e_samples_batches = [e_samples for _ in range(
                len(edge_samples_split[0]))]
        else:
            e_samples_batches = e_samples.split(batch_size)
        edge_samples_split.append(e_samples_batches)

    edge_samples_batches = []
    for i in range(len(edge_samples_split[0])):
        edge_samples_batches.append(
            [e_samples[i] for e_samples in edge_samples_split])

    return edge_samples_batches


def subpixel_sample(width, height, spp=1, sample_range=[]):
    if spp == 1:
        y_coords, x_coords = torch.meshgrid(torch.arange(
            height), torch.arange(width), indexing='ij')
    else:
        if len(sample_range) == 0:
            y_coords, x_coords = torch.meshgrid(torch.linspace(0, spp * height - 1, spp * height),
                                                torch.linspace(0, spp * width - 1, spp * width), indexing='ij')
            y_coords = y_coords / spp
            x_coords = x_coords / spp
        else:
            y_coords, x_coords = torch.meshgrid(torch.linspace(spp * sample_range[2], spp * sample_range[3] - 1,
                                                               spp * (sample_range[3] - sample_range[2])),
                                                torch.linspace(spp * sample_range[0], spp * sample_range[1] - 1,
                                                               spp * (sample_range[1] - sample_range[0])), indexing='ij')
            y_coords = y_coords / spp
            x_coords = x_coords / spp

    y_coords = y_coords.reshape(-1, 1)
    x_coords = x_coords.reshape(-1, 1)

    # Offset by 0.5 x bin
    y_coords = y_coords.float() + 1.0 / (2 * spp)
    x_coords = x_coords.float() + 1.0 / (2 * spp)

    y_coords = y_coords / height
    x_coords = x_coords / width

    # bounds assertion below ensures normalized coordinates

    assert y_coords.min() >= 0 and y_coords.max(
    ) <= 1 and x_coords.min() >= 0 and x_coords.max() <= 1

    samples = torch.hstack([y_coords, x_coords])

    return samples


def important_vertices(mlp, threshold=0.01):
    # L1 of slopes
    ve_features_extended = torch.clamp(mlp.ve_features[:, ...], min=0)

    w = mlp.w[ve_features_extended[..., 0]]
    l_fea = mlp.features[ve_features_extended[..., 1]]
    r_fea = mlp.features[ve_features_extended[..., 2]]

    # Blend features per corner using its two adjacent corners and the corresponding alpha values
    w = mlp.get_w(w, mlp.w_mask[ve_features_extended[..., 0]])
    w_padded = torch.where(
        mlp.ve_features[..., 0].unsqueeze(-1) >= 0, w, 0)
    if 'simple_fea' in mlp.opt_config and mlp.opt_config['simple_fea']:
        slope = w_padded
    else:
        slope = w_padded * (l_fea - r_fea)
    slope_max = slope.abs().max(dim=-1).values

    eid_pack = mlp.ve_features[..., 0].ravel()
    slope_max = slope_max.ravel()
    slope_max = slope_max[eid_pack >= 0]
    eid_pack = eid_pack[eid_pack >= 0]

    e_max = torch.zeros(
        mlp.mesh_e.shape[0], dtype=torch.float32, device=w.device)
    scatter_max(src=slope_max, index=eid_pack.long(), dim=0, out=e_max)

    w_mask = e_max.unsqueeze(-1) > threshold

    # Update bias of now semi-continuous or continuous vertices
    v_continuous_edges = (~w_mask[ve_features_extended[..., 0]]).int()
    v_continuous_edges_count = torch.where(
        mlp.ve_features[..., 0].unsqueeze(-1) >= 0, v_continuous_edges, 0)
    v_discontinuous = v_continuous_edges_count.squeeze(-1).sum(
        dim=-1) != (mlp.ve_features[..., 0] >= 0).sum(dim=-1)

    vid = torch.nonzero(v_discontinuous, as_tuple=False).squeeze()

    return vid


@ torch.no_grad()
def edge_sample(mlp, num_samples, importance_threshold=-1):
    v = mlp.mesh.get_v()
    e = mlp.mesh_e
    if e.device != v.device:
        e = e.to(v.device)
    e = e.long()

    # Filter boundary edges
    boundary_e = set(tuple(sorted(e))
                     for loop in mlp.mesh.boundary_edges for e in loop)
    boundary_e = torch.tensor(sorted(list(boundary_e)), device=v.device)
    boundary_e = boundary_e.long()
    boundary_mask = (
        (mlp.mesh_e.shape[0] * e[:, 0] + e[:, 1])[:, None] == mlp.mesh_e.shape[0] * boundary_e[:, 0] + boundary_e[:, 1]).any(-1)
    ei = torch.nonzero(~boundary_mask).squeeze()
    assert ei.shape[0] == mlp.mesh_e.shape[0] - boundary_e.shape[0]

    # Compute the edge lengths
    ee = v[e[:, 1], :] - v[e[:, 0], :]
    ee_normal = torch.zeros_like(ee)
    ee_normal[:, 0] = -ee[:, 1]
    ee_normal[:, 1] = ee[:, 0]
    ee_length = torch.norm(ee, dim=1)
    ee_length = ee_length.unsqueeze(-1)
    ee_normal = ee_normal / ee_length

    # Generate the samples
    samples = torch.rand(num_samples, 1, device=v.device)

    # Sample edge
    if importance_threshold > 0:
        vid = important_vertices(mlp, threshold=importance_threshold)
        samples_ei = []
        for i in vid:
            samples_ei.append(torch.nonzero((e[ei] == i).any(-1)))

        e_prob = torch.ones((ei.shape[0], 1), dtype=v.dtype, device=v.device)
        if len(samples_ei) > 0:
            samples_ei = torch.vstack(samples_ei)
            samples_ei = samples_ei.unique()
            e_prob[samples_ei] = 5
        e_prob = e_prob / e_prob.sum()

        imp_samples = torch.multinomial(
            e_prob.ravel(), num_samples=num_samples, replacement=True)
        samples_ei = ei[imp_samples]
        e_prob = e_prob[imp_samples]
    else:
        samples_ei = torch.randint(
            low=0, high=ei.shape[0], size=(num_samples,)).to(v.device)
        samples_ei = ei[samples_ei]
        num_int_edges = ei.shape[0]
        e_prob = torch.ones(
            (samples_ei.shape[0], 1), dtype=v.dtype, device=v.device) / num_int_edges

    # debug-only edge selection removed

    return samples, samples_ei, ee_normal, ee_length, e_prob


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def grid_sample_triangle(v, f, fid_in, c_offset=-1e3, step_size=1.0, pixel_aligned=False):
    fid = fid_in
    if not is_iterable(fid_in):
        fid = [fid_in]

    sample_start_time = time.time()

    # 1. Get the AABB of the triangle
    ff = f[fid]
    ff = ff.flatten()
    p_min, p_max = v[ff, :].min(
        axis=0).values, v[ff, :].max(axis=0).values

    if pixel_aligned:
        p_min = torch.floor(p_min)

    sample_end_time = time.time()
    # timing logs suppressed by default

    # 2. Sample the AABB with step_size
    def grid_sample_aabb(p_min, p_max, step_size):
        # Create a grid of coordinates using meshgrid
        x_range = np.arange(p_min[0], p_max[0] + step_size, step_size)
        y_range = np.arange(p_min[1], p_max[1] + step_size, step_size)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Stack the grid coordinates to get a list of points
        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Filter the points to keep only those within the AABB
        sampled_points = points[(points[:, 0] >= p_min[0].numpy()) & (points[:, 0] <= p_max[0].numpy()) & (
            points[:, 1] >= p_min[1].numpy()) & (points[:, 1] <= p_max[1].numpy())]

        return sampled_points

    sample_start_time = time.time()
    sampled_points = grid_sample_aabb(p_min.cpu(), p_max.cpu(), step_size)
    sample_end_time = time.time()
    # print(
    #     f"\t\tgrid_sample_aabb Time: {(sample_end_time - sample_start_time):.4f} seconds")

    # 3. Check if the sample is inside the triangle. Reject if not.
    ray_start_time = time.time()
    c = np.hstack((sampled_points, c_offset *
                   np.ones((sampled_points.shape[0], 1))))
    d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
    _, ids, _ = ray_mesh_intersect(c, d, v.cpu().numpy(), f.cpu().numpy())
    ray_end_time = time.time()
    # print(
    #     f"\t\tray_mesh_intersect Time: {(ray_end_time - ray_start_time):.4f} seconds")

    start_time = time.time()
    # valid_flags = ids == fid
    valid_mask = np.isin(ids, fid)
    in_ids = ids[valid_mask]
    c = c[valid_mask, :]
    c[:, 2] = 0
    end_time = time.time()
    # print(
    #     f"\t\tafter ray_mesh_intersect Time: {(end_time - start_time):.4f} seconds")

    return c, in_ids


def grid_sample_triangle_aabb(v, f, fid_in, p_min, p_max, c_offset=-1e3, step_size=1.0):
    fid = fid_in
    if not is_iterable(fid_in):
        fid = [fid_in]

    # 2. Sample the AABB with step_size
    def grid_sample_aabb(p_min, p_max, step_size):
        # Create a grid of coordinates using meshgrid
        x_range = np.arange(p_min[0], p_max[0] + step_size, step_size)
        y_range = np.arange(p_min[1], p_max[1] + step_size, step_size)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Stack the grid coordinates to get a list of points
        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Filter the points to keep only those within the AABB
        sampled_points = points[(points[:, 0] >= p_min[0].numpy()) & (points[:, 0] <= p_max[0].numpy()) & (
            points[:, 1] >= p_min[1].numpy()) & (points[:, 1] <= p_max[1].numpy())]

        return sampled_points

    sample_start_time = time.time()
    sampled_points = grid_sample_aabb(p_min, p_max, step_size)
    sample_end_time = time.time()
    # print(
    #     f"\t\tgrid_sample_aabb Time: {(sample_end_time - sample_start_time):.4f} seconds")

    # 3. Check if the sample is inside the triangle. Reject if not.
    ray_start_time = time.time()
    c = np.hstack((sampled_points, c_offset *
                   np.ones((sampled_points.shape[0], 1))))
    d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
    _, ids, _ = ray_mesh_intersect(c, d, v.cpu().numpy(), f.cpu().numpy())
    ray_end_time = time.time()
    # print(
    #     f"\t\tray_mesh_intersect Time: {(ray_end_time - ray_start_time):.4f} seconds")

    start_time = time.time()
    # valid_flags = ids == fid
    valid_mask = np.isin(ids, fid)
    in_ids = ids[valid_mask]
    c = c[valid_mask, :]
    c[:, 2] = 0
    end_time = time.time()
    # print(
    #     f"\t\tafter ray_mesh_intersect Time: {(end_time - start_time):.4f} seconds")

    return c, in_ids


def sample_triangle_stratified(v, f, fid_in, step_size=1, min_samples=16):
    # Estimate the number of samples needed to cover the triangle
    # 1. Get the AABB of the triangle
    fid = fid_in
    if not is_iterable(fid_in):
        fid = [fid_in]

    # 1. Compute #samples per triangle
    ff = f[fid]
    # p_min, p_max = v[:, 0:2][f[fid], :].min(
    #     axis=1).values, v[:, 0:2][f[fid], :].max(axis=1).values

    # pp = (p_max - p_min).abs()
    # num_samples = (pp[:, 0] * pp[:, 1] / step_size).ceil().int()

    vf = v[ff]
    e1 = vf[:, 1, :] - vf[:, 0, :]
    e2 = vf[:, 2, :] - vf[:, 0, :]
    face_normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(face_normals, p=2, dim=1)
    num_samples = torch.ceil(areas / step_size).int()
    num_samples = torch.clamp(num_samples, min=min_samples)

    # 2. Generate random 2D samples
    samples = torch.rand(num_samples.sum(), 2, device=v.device)
    in_faces = torch.cat([torch.full((count,), i, dtype=torch.long)
                          for i, count in enumerate(num_samples)]).to(v.device)

    # 3. Map the samples to the triangle
    # https://pharr.org/matt/blog/2019/02/27/triangle-sampling-1
    su0 = torch.sqrt(samples[:, 0])
    b0 = 1 - su0
    b1 = samples[:, 1] * su0
    b2 = 1 - b0 - b1

    # 4. Map the barycentric coordinates to the triangle
    p = b0.unsqueeze(-1) * v[f[in_faces][:, 0], :] + b1.unsqueeze(-1) * \
        v[f[in_faces][:, 1], :] + b2.unsqueeze(-1) * v[f[in_faces][:, 2], :]

    return p, in_faces


def bary_sample_triangle_stratified(v, f, step_size=1, min_samples=16):
    # Estimate the number of samples needed to cover the triangle
    # 1. Compute #samples per triangle
    ff = f

    vf = v[ff]
    e1 = vf[:, 1, :] - vf[:, 0, :]
    e2 = vf[:, 2, :] - vf[:, 0, :]
    face_normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(face_normals, p=2, dim=1)
    num_samples = torch.ceil(areas / step_size).int()
    num_samples = torch.clamp(num_samples, min=min_samples)

    # 2. Generate random 2D samples
    samples = torch.rand(num_samples.sum(), 2, device=v.device)
    in_faces = torch.cat([torch.full((count,), i, dtype=torch.long)
                          for i, count in enumerate(num_samples)]).to(v.device)
    # Face to pixels
    px_positions = torch.arange(in_faces.shape[0], device=v.device)
    sorted_vals, sorted_indices = torch.sort(in_faces)
    sorted_px_positions = px_positions[sorted_indices]
    counts = torch.bincount(in_faces)
    f2px = list(torch.split(sorted_px_positions, counts.tolist()))
    for i in range(f.shape[0] - len(f2px)):
        f2px.append(torch.tensor([], device=v.device))
    f2px = torch.nn.utils.rnn.pad_sequence(
        f2px, batch_first=True, padding_value=-1)

    # 3. Map the samples to the triangle
    # https://pharr.org/matt/blog/2019/02/27/triangle-sampling-1
    su0 = torch.sqrt(samples[:, 0])
    b0 = 1 - su0
    b1 = samples[:, 1] * su0
    b2 = 1 - b0 - b1

    # 4. Map the barycentric coordinates to the triangle
    # p = b0.unsqueeze(-1) * v[f[in_faces][:, 0], :] + b1.unsqueeze(-1) * \
    #     v[f[in_faces][:, 1], :] + b2.unsqueeze(-1) * v[f[in_faces][:, 2], :]
    b_samples = torch.stack([b0, b1, b2], dim=1)

    return b_samples, in_faces, f2px


def sample_triangle_uniform(v, f, fid_in, step_size=1, min_samples=16):
    # Osada, R., Funkhouser, T., Chazelle, B., & Dobkin, D. (2002). Shape distributions. ACM Transactions on Graphics (TOG), 21(4), 807-832.
    # Estimate the number of samples needed to cover the triangle
    # 1. Get the AABB of the triangle
    fid = fid_in
    if not is_iterable(fid_in):
        fid = [fid_in]

    # 1. Get the AABB of the triangle
    ff = f[fid]
    ff = ff.flatten()
    p_min, p_max = v[ff, :].min(
        axis=0).values, v[ff, :].max(axis=0).values

    pp = p_max - p_min
    num_samples = int(pp[0] * pp[1] / step_size)
    num_samples = max(num_samples, min_samples)

    # https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain
    def trisample(A, B, C):
        """
        Given three vertices A, B, C,
        sample point uniformly in the triangle
        """
        r1 = torch.rand(A.shape[0], 1)
        r2 = torch.rand(A.shape[0], 1)

        s1 = torch.sqrt(r1)

        samples = A * (1.0 - s1) + B * (1.0 - r2) * s1 + C * r2 * s1

        return samples

    samples = trisample(v[f[fid][:, 0], :],
                        v[f[fid][:, 1], :], v[f[fid][:, 2], :])
    return samples
