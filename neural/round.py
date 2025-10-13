import time
import gc

import pickle
import torch
from torch_scatter import scatter_max
import numpy as np

from gpytoolbox import ray_mesh_intersect
from tools.geometry_utils import compute_angles_atan2
from learning.edge_sampling_render import (
    monte_carlo_interior_render_samples, sum_rendering)
from tools.utils import temporary_directory
from tools.utils import load_mlp

sample_color_cache = []
f2px = None


@torch.no_grad()
def compute_offset(mlp):
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

    return e_max


@torch.no_grad()
def discard_edges(mlp, w_mask):
    # L1 of slopes
    ve_features_extended = torch.clamp(mlp.ve_features[:, ...], min=0)

    w = mlp.w[ve_features_extended[..., 0]]
    l_fea = mlp.features[ve_features_extended[..., 1]]
    r_fea = mlp.features[ve_features_extended[..., 2]]

    # Blend features per corner using its two adjacent corners and the corresponding alpha values
    w = mlp.get_w(w, mlp.w_mask[ve_features_extended[..., 0]])
    w_padded = torch.where(
        mlp.ve_features[..., 0].unsqueeze(-1) >= 0, w, 0)

    # Update bias of now semi-continuous or continuous vertices
    v_continuous_edges = (~w_mask[ve_features_extended[..., 0]]).int()
    v_continuous_edges_count = torch.where(
        mlp.ve_features[..., 0].unsqueeze(-1) >= 0, v_continuous_edges, 0)
    # v_continuous = v_continuous_edges_count.squeeze(-1).sum(
    #     dim=-1) == (mlp.ve_features[..., 0] >= 0).sum(dim=-1)
    v_continuous = v_continuous_edges_count.squeeze(-1).sum(dim=-1) > 0
    vid = torch.nonzero(v_continuous, as_tuple=False).squeeze()

    v_offset = 1e-4 * torch.ones((1, 3), device=mlp.mesh.v.device)
    v_offset[:, 2] = 0

    v = mlp.get_v().detach()
    valid_v = v[vid]
    cc = valid_v + v_offset

    # Convert sample points to theta coordinates
    # Compute the angle wrt the reference edge
    # |S| x |v| (3x|f|) x MaxAdjE
    v2v_extended = torch.clamp(mlp.v2v[vid], min=0)
    # |S| x |v| (3x|f|) x 3
    dir_ref = v[v2v_extended[..., 0]] - valid_v
    dir_thetas = cc - valid_v

    # TODO: Handle the degenerate case of hitting a vertex exactly

    # Compute the rotation angles
    # Calculate CCW angle
    # angle_ccw = compute_angles(dir_ref, dir_thetas)
    angle_ccw = compute_angles_atan2(dir_ref, dir_thetas)
    # |S| x |v| (3x|f|) x |theta|
    thetas = mlp.thetas[vid]
    c_normalized = (angle_ccw.unsqueeze(-1) - thetas) / (2 * torch.pi)

    # |S| x |v| (3x|f|) x |theta|
    # Different thetas define different coordinate systems
    t = torch.fmod(c_normalized + 1, 1)

    ve_features_extended = torch.clamp(mlp.ve_features[vid], min=0)

    w = mlp.w[ve_features_extended[..., 0]]
    l_fea = mlp.features[ve_features_extended[..., 1]]
    r_fea = mlp.features[ve_features_extended[..., 2]]
    bias = mlp.bias[vid]

    # # Blend features per corner using its two adjacent corners and the corresponding alpha values
    w = mlp.get_w(w, mlp.w_mask[ve_features_extended[..., 0]])
    w_padded = torch.where(
        mlp.ve_features[vid][..., 0].unsqueeze(-1) >= 0, w, 0)
    # fea_thetas = w_padded * r_fea
    # Or if we don't ignore the supposed-to-be small term w_padded * t.unsqueeze(-1)
    # fea_thetas = w_padded * r_fea + w_padded * t.unsqueeze(-1)
    # Or we shift to the mid point
    fea_thetas = w_padded * r_fea + w_padded * (l_fea - r_fea) * 0.5

    continuous_edge_mask = (~w_mask[ve_features_extended[..., 0]]) & (
        mlp.ve_features[vid, ..., 0].unsqueeze(-1) >= 0)
    for i in range(fea_thetas.shape[2]):
        fea_thetas[..., i][~continuous_edge_mask.squeeze(-1)] = 0
    fea = fea_thetas.sum(1) + bias

    mlp.w[ve_features_extended[...,
                               0].unsqueeze(-1)[continuous_edge_mask]] = -float('inf')
    mlp.bias[vid] = fea

    mlp.w_mask = torch.logical_and(mlp.w_mask, w_mask)

    return mlp


@torch.no_grad()
def threshold_w(mlp, threshold=0.1):
    e_max = compute_offset(mlp)

    w_mask = e_max.unsqueeze(-1) > threshold
    mlp = discard_edges(mlp, w_mask)

    return mlp


@torch.no_grad()
def test_render_cached(mlp, x, int_samples_batches, update_faces):
    global sample_color_cache, f2px

    int_samples_, sample_ids_ = int_samples_batches

    # Pick samples that are not cached
    v = mlp.mesh.get_v()

    f2px_dummy = torch.vstack(
        [-1 * torch.ones_like(f2px[0, :]).unsqueeze(0), f2px]).to(v.device).to(torch.float32)
    f2px_dummy[f2px_dummy < 0] = torch.nan
    f_k_ring2 = torch.tensor(update_faces, device=v.device)
    px_all = f2px_dummy[f_k_ring2 + 1].reshape(-1, 1)
    px_all = px_all[~torch.isnan(px_all)]
    px_mask = torch.zeros_like(sample_ids_).bool()
    px_mask[px_all.long()] = True

    # start_time = time.time()

    samples_img = x * \
        torch.tensor([mlp.image.size[1],
                      mlp.image.size[0]], device=mlp.mesh.device)
    samples_pixel = torch.floor(samples_img).int()
    samples_pixel[:, 0] = torch.clamp(
        samples_pixel[:, 0], 0, mlp.image.size[1] - 1)
    samples_pixel[:, 1] = torch.clamp(
        samples_pixel[:, 1], 0, mlp.image.size[0] - 1)

    int_rendering = torch.zeros([mlp.image.size[0] * mlp.image.size[1], mlp.layers[-1].out_features],
                                dtype=torch.float32, device=mlp.device)
    int_spp = torch.zeros([mlp.image.size[0] * mlp.image.size[1]],
                          dtype=mlp.mesh.f.dtype, device=mlp.device)
    colors_accum = []
    samples_accum = []
    test_timing = False
    batch_idx = 0

    # Read cached samples
    sample_ids_cached = sample_ids_[~px_mask]
    int_colors_cached = sample_color_cache[sample_ids_cached]
    if int_colors_cached.shape[0] > 0:
        colors_accum.append(int_colors_cached)
        samples_accum.append(int_samples_[~px_mask])

    batch_size = 4 * 256 * 1024
    int_samples_batches_updates = int_samples_[px_mask].split(batch_size)
    sample_ids_batches_updates = sample_ids_[px_mask].split(batch_size)
    for int_samples, sample_ids in zip(int_samples_batches_updates, sample_ids_batches_updates):
        # Look up the cache

        # torch.cuda.synchronize()
        # b_start_time = time.time()

        int_colors, int_samples_valid = monte_carlo_interior_render_samples(
            mlp, int_samples)
        colors_accum.append(int_colors)
        samples_accum.append(int_samples_valid)

        assert int_samples_valid.shape[0] == sample_ids.shape[0]

        # Write to cache
        sample_color_cache[sample_ids] = int_colors

        # torch.cuda.synchronize()
        # b_end_time = time.time()
        # if test_timing:
        #     b_execution_time = b_end_time - b_start_time
        #     print(
        #         f'\tAccum interior batch rendering time: {b_execution_time:.4f} s')
        #     test_timing = False

        batch_idx += 1

    colors_accum = torch.vstack(colors_accum)
    samples_accum = torch.vstack(samples_accum)
    int_rendering, int_spp = sum_rendering(
        mlp, colors_accum, samples_accum, flip_axis=True)

    int_rendering = int_rendering / \
        torch.clamp(int_spp, min=1).unsqueeze(-1)

    # torch.cuda.synchronize()
    # end_time = time.time()
    # int_execution_time = end_time - start_time
    # print(
    #     f'Interior rendering time: {int_execution_time:.4f} s')

    # Actually compute the loss
    samples_indices = samples_pixel[:, 0] * \
        mlp.image.size[0] + samples_pixel[:, 1]
    y_hat = torch.gather(
        int_rendering, 0, samples_indices.view(-1, 1).long().expand(-1, int_rendering.shape[-1]))

    return y_hat


@torch.no_grad()
def test_render(mlp, x, int_samples_batches):
    start_time = time.time()

    samples_img = x * \
        torch.tensor([mlp.image.size[1],
                      mlp.image.size[0]], device=mlp.mesh.device)
    samples_pixel = torch.floor(samples_img).int()
    samples_pixel[:, 0] = torch.clamp(
        samples_pixel[:, 0], 0, mlp.image.size[1] - 1)
    samples_pixel[:, 1] = torch.clamp(
        samples_pixel[:, 1], 0, mlp.image.size[0] - 1)

    int_rendering = torch.zeros([mlp.image.size[0] * mlp.image.size[1], mlp.layers[-1].out_features],
                                dtype=torch.float32, device=mlp.device)
    int_spp = torch.zeros([mlp.image.size[0] * mlp.image.size[1]],
                          dtype=mlp.mesh.f.dtype, device=mlp.device)
    colors_accum = []
    samples_accum = []
    test_timing = False
    batch_idx = 0
    for int_samples in int_samples_batches:
        # torch.cuda.synchronize()
        b_start_time = time.time()

        int_colors, int_samples_valid = monte_carlo_interior_render_samples(
            mlp, int_samples)
        colors_accum.append(int_colors)
        samples_accum.append(int_samples_valid)

        # torch.cuda.synchronize()
        b_end_time = time.time()
        if test_timing:
            b_execution_time = b_end_time - b_start_time
            # print(
            #     f'\tAccum interior batch rendering time: {b_execution_time:.4f} s')
            test_timing = False

        batch_idx += 1

    colors_accum = torch.vstack(colors_accum)
    samples_accum = torch.vstack(samples_accum)
    int_rendering, int_spp = sum_rendering(
        mlp, colors_accum, samples_accum, flip_axis=True)

    int_rendering = int_rendering / \
        torch.clamp(int_spp, min=1).unsqueeze(-1)

    # torch.cuda.synchronize()
    end_time = time.time()
    int_execution_time = end_time - start_time
    # print(
    #     f'Interior rendering time: {int_execution_time:.4f} s')

    # Actually compute the loss
    samples_indices = samples_pixel[:, 0] * \
        mlp.image.size[0] + samples_pixel[:, 1]
    y_hat = torch.gather(
        int_rendering, 0, samples_indices.view(-1, 1).long().expand(-1, int_rendering.shape[-1]))

    return y_hat


@torch.no_grad()
def adaptive_round_w(mlp, samples, int_samples_batches, inc_ratio=0.1):
    x, y = samples
    e_max = compute_offset(mlp)
    sorted, e_indices = torch.sort(e_max)
    cur_w_mask = e_max.unsqueeze(-1).clone()
    cur_w_mask = cur_w_mask.fill_(1).bool()
    cur_w_mask = torch.logical_and(mlp.w_mask, cur_w_mask)

    y_hat = test_render(mlp, x, int_samples_batches)
    loss_func = torch.nn.MSELoss()
    loss_ref = loss_func(y_hat, y)

    # inc_ratio = min(inc_ratio, loss_ref * 0.01)
    with temporary_directory() as tmp_dir:
        pickle_file_path = tmp_dir / 'model_copy.pkl'
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(mlp, pickle_file)
        e_indices_dense = []
        for i in range(e_indices.shape[0]):
            if not cur_w_mask[e_indices[i]]:
                continue
            e_indices_dense.append(e_indices[i])
        for i in range(len(e_indices_dense)):
            # print(f'Iteration {i} / {len(e_indices_dense)}')
            cur_w_mask[e_indices_dense[i]] = False

            # Save the current state
            mlp_test = load_mlp(pickle_file_path)
            mlp_test = discard_edges(mlp_test, cur_w_mask)

            # Render
            y_hat = test_render(mlp_test, x, int_samples_batches)
            loss = loss_func(y_hat, y)
            del mlp_test
            gc.collect()
            torch.cuda.empty_cache()

            # loss_change = loss - loss_ref / loss_ref
            loss_change = loss - loss_ref
            if float(loss_change) > inc_ratio:
                # print(
                #     f'Edge {e_indices_dense[i]} kept, loss change: {loss_change} = {loss} / {loss_ref}')
                cur_w_mask[e_indices_dense[i]] = True
            # else:
            #     print(
            #         f'Edge {e_indices_dense[i]} discarded, loss change: {loss_change} = {loss} / {loss_ref}')

            if float(loss_change) > 1.5 * inc_ratio:
                break

    return discard_edges(mlp, cur_w_mask)


@torch.no_grad()
def round_w(mlp, samples, int_samples_batches, threshold=0.1, inc_ratio=0.1):
    mlp = threshold_w(mlp, threshold=threshold)
    mlp = adaptive_round_w(
        mlp, samples, int_samples_batches, inc_ratio=inc_ratio)

    return mlp


@torch.no_grad()
def adaptive_round_w_cached(mlp, samples, int_samples_batches, inc_ratio=0.1):
    global sample_color_cache

    int_samples, sample_ids = int_samples_batches

    x, y = samples
    e_max = compute_offset(mlp)
    sorted, e_indices = torch.sort(e_max)
    cur_w_mask = e_max.unsqueeze(-1).clone()
    cur_w_mask = cur_w_mask.fill_(1).bool()
    cur_w_mask = torch.logical_and(mlp.w_mask, cur_w_mask)

    y_hat = test_render_cached(
        mlp, x, [int_samples, sample_ids], update_faces=list(range(mlp.mesh.f.shape[0])))
    loss_func = torch.nn.MSELoss()
    loss_ref = loss_func(y_hat, y)

    # inc_ratio = min(inc_ratio, loss_ref * 0.01)
    calibrate_itr = 500
    with temporary_directory() as tmp_dir:
        pickle_file_path = tmp_dir / 'model_copy.pkl'
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(mlp, pickle_file)
        e_indices_dense = []
        for i in range(e_indices.shape[0]):
            if not cur_w_mask[e_indices[i]]:
                continue
            e_indices_dense.append(e_indices[i])
        for i in range(len(e_indices_dense)):
            # if i > 10:
            #     exit()
            # print(f'Iteration {i} / {len(e_indices_dense)}')
            cur_w_mask[e_indices_dense[i]] = False

            # Save the current state
            mlp_test = load_mlp(pickle_file_path)
            sample_color_cache_saved = sample_color_cache.clone()

            # Find the neighborhood
            v0 = mlp.mesh_e[e_indices_dense[i], 0]
            v1 = mlp.mesh_e[e_indices_dense[i], 1]
            ff = torch.cat([mlp.mesh.vf[v0], mlp.mesh.vf[v1]], dim=0)
            ff = ff[ff >= 0]
            ff = ff.unique()

            # Discard and render
            mlp_test = discard_edges(mlp_test, cur_w_mask)
            y_hat = test_render_cached(mlp_test, x, [
                                       int_samples, sample_ids], update_faces=ff.tolist())
            loss = loss_func(y_hat, y)

            # loss_change = loss - loss_ref / loss_ref
            loss_change = loss - loss_ref
            if float(loss_change) > inc_ratio:
                # print(
                #     f'Edge {e_indices_dense[i]} kept, loss change: {loss_change} = {loss} / {loss_ref}')
                cur_w_mask[e_indices_dense[i]] = True
                sample_color_cache = sample_color_cache_saved
            else:
                if i % calibrate_itr == 0 and i > 0:
                    test_render_cached(mlp_test, x, [int_samples, sample_ids], update_faces=list(
                        range(mlp.mesh.f.shape[0])))
                # print(
                #     f'Edge {e_indices_dense[i]} discarded, loss change: {loss_change} = {loss} / {loss_ref}')

            del mlp_test
            gc.collect()
            torch.cuda.empty_cache()

            if float(loss_change) > 1.5 * inc_ratio:
                break

    return discard_edges(mlp, cur_w_mask)


@torch.no_grad()
def round_w_cached(mlp, samples, int_samples, threshold=0.1, inc_ratio=0.1):
    global f2px, sample_color_cache

    c = np.column_stack(
        (int_samples.detach().cpu().numpy(), -1 * np.ones(int_samples.shape[0])))
    c[:, [0, 1]] = c[:, [1, 0]]
    c[:, 0] *= mlp.mesh.size[0]
    c[:, 1] *= mlp.mesh.size[1]
    c[:, 2] = -1
    d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
    _, ids, _ = ray_mesh_intersect(
        c, d, mlp.mesh.get_v().detach().cpu().numpy(), mlp.mesh.f.cpu().numpy())

    valid_flags = ids >= 0
    int_samples = int_samples[valid_flags]
    in_faces = torch.from_numpy(ids[valid_flags]).to(mlp.device)

    px_positions = torch.arange(in_faces.shape[0], device=mlp.mesh.v.device)
    sorted_vals, sorted_indices = torch.sort(in_faces)
    sorted_px_positions = px_positions[sorted_indices]
    counts = torch.bincount(in_faces)
    f2px = list(torch.split(sorted_px_positions, counts.tolist()))
    for i in range(mlp.mesh.f.shape[0] - len(f2px)):
        f2px.append(torch.tensor([], device=mlp.v.device))
    f2px = torch.nn.utils.rnn.pad_sequence(
        f2px, batch_first=True, padding_value=-1)
    int_samples_batches = (int_samples, torch.arange(
        int_samples.shape[0], device=mlp.mesh.v.device))

    # Initialize cache
    if len(sample_color_cache) == 0:
        sample_color_cache = torch.zeros([int_samples.shape[0], mlp.layers[-1].out_features],
                                         dtype=torch.float32, device=mlp.device)

    mlp = threshold_w(mlp, threshold=threshold)
    mlp = adaptive_round_w_cached(
        mlp, samples, int_samples_batches, inc_ratio=inc_ratio)

    return mlp
