import time

import igl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_scatter import scatter_mean, scatter_sum
from gpytoolbox import ray_mesh_intersect

from geometry.defgrid import find_k_ring_vf
from geometry.softras import is_point_in_triangle_mat, point_triangle_squared_distance_mat
from geometry.mesh_triangle import TriangleMesh
from learning.nerf2d_data import prepare_data
from tools.plot_utils import plot_association_func, plot_mesh, to_pil_image
from tools.param import epsilon
from learning.sampler import (
    grid_sample_triangle, grid_sample_triangle_aabb, sample_triangle_stratified, bary_sample_triangle_stratified)


@torch.no_grad()
def face_mean_color(mesh: TriangleMesh, image, samples_, in_faces, is_barycentric=False):
    v = mesh.get_v()
    if is_barycentric:
        samples = samples_[:, 0].unsqueeze(-1) * v[mesh.f[in_faces][:, 0], :] + \
            samples_[:, 1].unsqueeze(-1) * v[mesh.f[in_faces][:, 1], :] + \
            samples_[:, 2].unsqueeze(-1) * v[mesh.f[in_faces][:, 2], :]
    else:
        samples = samples_

    samples_pixel = torch.floor(samples).int()

    # Compute colors
    # |F| x 3
    # Note that we use the corresponding integer coordinates for the sample points (which are all 0.5-offset)
    image_array = np.array(image)
    if len(image_array.shape) == 2:
        image_array = image_array[..., np.newaxis]
    samples_pixel[:, 0] = torch.clamp(
        samples_pixel[:, 0], 0, image.width - 1)
    samples_pixel[:, 1] = torch.clamp(
        samples_pixel[:, 1], 0, image.height - 1)
    C = torch.from_numpy(
        image_array[samples_pixel[:, 1].cpu().int(), samples_pixel[:, 0].cpu().int()] / 255.0).type(mesh.v.dtype)
    C = C.to(mesh.device)

    C_mean_all = torch.zeros(
        (mesh.f.shape[0], C.shape[-1]), dtype=mesh.v.dtype, device=mesh.device)
    for d in range(C.shape[-1]):
        scatter_mean(C[:, d], in_faces.long(), out=C_mean_all[:, d])

    # scatter_mean(C[:, 0], in_faces.long(), out=C_mean_all[:, 0])
    # scatter_mean(C[:, 1], in_faces.long(), out=C_mean_all[:, 1])
    # scatter_mean(C[:, 2], in_faces.long(), out=C_mean_all[:, 2])
    C_mean = C_mean_all[0:mesh.f_interior_count, :]

    return C_mean


@torch.no_grad()
def compute_sample_weights(v, f, in_faces):
    int_spp = torch.zeros([f.shape[0]],
                          dtype=f.dtype, device=v.device)
    int_counts = torch.ones_like(in_faces).to(f.dtype)
    scatter_sum(src=int_counts.ravel(),
                index=in_faces.long().ravel(), out=int_spp)
    assert (int_spp[in_faces] > 0).all()
    sample_weights = 1 / int_spp[in_faces]

    vf = v[f]
    e1 = vf[:, 1, :] - vf[:, 0, :]
    e2 = vf[:, 2, :] - vf[:, 0, :]
    face_normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(face_normals, p=2, dim=1)

    sample_weights = sample_weights * areas[in_faces]

    return sample_weights


@torch.no_grad()
def prepare_face_samples(mesh: TriangleMesh, image, k_ring, loss_mask=None):
    face_mask = list(range(mesh.f.shape[0]))
    if isinstance(loss_mask, torch.Tensor):
        face_mask = loss_mask.nonzero().squeeze().cpu().numpy().tolist()
    face_mask_set = set(face_mask)

    v = mesh.get_v()
    samples, in_faces = sample_triangle_stratified(
        v, mesh.f, face_mask, step_size=1, min_samples=10)

    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).to(mesh.device)
    if isinstance(in_faces, np.ndarray):
        in_faces = torch.from_numpy(in_faces).to(mesh.device)

    # Compute sample weights based on pixel
    sample_weights = compute_sample_weights(v, mesh.f, in_faces)

    # Compute sample-face association weights
    # Find the K-ring of the triangle for efficient computation
    # We assume a triangle is only to be deformed slightly
    # so the variance computation can be local
    if len(mesh.tt) == 0:
        with torch.no_grad():
            mesh.tt, _ = igl.triangle_triangle_adjacency(mesh.f.cpu().numpy())
    if not hasattr(mesh, 'vf') or len(mesh.vf) == 0:
        with torch.no_grad():
            vf, ni = igl.vertex_triangle_adjacency(
                mesh.f.cpu().numpy(), mesh.v.shape[0])

            # Initialize the result array with -1 for padding
            max_adjacent_faces = max(ni[1:] - ni[:-1])
            mesh.vf = np.full((mesh.v.shape[0], max_adjacent_faces), -1)

            # Iterate over each vertex
            for i in range(mesh.v.shape[0]):
                adjacent_faces = vf[ni[i]:ni[i + 1]]
                mesh.vf[i, :len(adjacent_faces)] = adjacent_faces
            mesh.vf = torch.from_numpy(mesh.vf).int().to(mesh.device)

    # Fetch the neighborhood per triangle per sample
    # Use the largest neighborhood size to make a dense but efficient matrix
    max_neighborhood_size = 0
    max_sample_size = 0
    p_indices = []

    k_ring_time = 0
    sample_lookup_time = 0
    save_time = 0
    kring_start_time = time.time()
    for fid in range(mesh.f_interior_count):
        if fid not in face_mask_set:
            continue
        k_f_start_time = time.time()

        # Use the k-ring as our domain
        # f_k_ring = find_k_ring(mesh.tt, fid, k_ring)
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

        save_time += time.time() - sample_lookup_end_time

    kring_end_time = time.time()
    kring_execution_time = kring_end_time - kring_start_time
    # timings suppressed by default

    p_indices = torch.nn.utils.rnn.pad_sequence(
        p_indices, batch_first=True, padding_value=-1)
    # Smax x |F|
    p_indices = p_indices.squeeze(-1).T
    p_indices_extended = p_indices >= 0
    sf_weights = p_indices_extended.sum(dim=0)
    assert (sf_weights > 0).all()
    sf_weights = 1 / sf_weights

    return samples, in_faces, sample_weights, sf_weights


@torch.no_grad()
def prepare_face_bary_samples(mesh: TriangleMesh):
    v = mesh.get_v()
    b_samples, in_faces, f2px = bary_sample_triangle_stratified(
        v, mesh.f, step_size=1, min_samples=10)
    mesh.f2px = f2px

    return b_samples, in_faces


def sample_softras(mesh: TriangleMesh, image, sigma, k_ring,
                   samples, in_faces, C_mean,
                   loss_mask=None, to_normalize=True):
    # Find the K-ring of the triangle for efficient computation
    # We assume a triangle is only to be deformed slightly
    # so the variance computation can be local
    if len(mesh.tt) == 0:
        with torch.no_grad():
            mesh.tt, _ = igl.triangle_triangle_adjacency(mesh.f.cpu().numpy())
    if not hasattr(mesh, 'vf') or len(mesh.vf) == 0:
        with torch.no_grad():
            vf, ni = igl.vertex_triangle_adjacency(
                mesh.f.cpu().numpy(), mesh.v.shape[0])

            # Initialize the result array with -1 for padding
            max_adjacent_faces = max(ni[1:] - ni[:-1])
            mesh.vf = np.full((mesh.v.shape[0], max_adjacent_faces), -1)

            # Iterate over each vertex
            for i in range(mesh.v.shape[0]):
                adjacent_faces = vf[ni[i]:ni[i + 1]]
                mesh.vf[i, :len(adjacent_faces)] = adjacent_faces
            mesh.vf = torch.from_numpy(mesh.vf).int().to(mesh.device)

    if not hasattr(mesh, 'f_k_ring') or len(mesh.f_k_ring) == 0:
        with torch.no_grad():
            # Build one-ring look up fast
            assert k_ring == 1
            fid = torch.arange(mesh.f.shape[0], device=mesh.device)
            vid = mesh.f[fid]
            f_k_ring = mesh.vf[vid].reshape(mesh.f.shape[0], -1)
            tmpt = torch.unbind(f_k_ring)
            mesh.f_k_ring = [torch.unique(t[t >= 0]) for t in tmpt]
            mesh.f_k_ring = torch.nn.utils.rnn.pad_sequence(
                mesh.f_k_ring, batch_first=True, padding_value=-1)

    # The actual positions
    v = mesh.get_v()

    # Find sample-face association
    # |S| x 3
    with torch.no_grad():
        face_mask = list(range(mesh.f.shape[0]))
        if isinstance(loss_mask, torch.Tensor):
            face_mask = loss_mask.nonzero().squeeze().cpu().numpy().tolist()
        face_mask_set = set(face_mask)

        # torch.cuda.synchronize()
        # new_start_time = time.time()

        # Face to pixels
        px_positions = torch.arange(in_faces.shape[0], device=mesh.device)
        _, sorted_indices = torch.sort(in_faces)
        sorted_px_positions = px_positions[sorted_indices]
        counts = torch.bincount(in_faces)
        f2px_ = list(torch.split(sorted_px_positions, counts.tolist()))
        for i in range(mesh.f.shape[0] - len(f2px_)):
            f2px_.append(torch.tensor([], device=mesh.device))
        f2px_ = torch.nn.utils.rnn.pad_sequence(
            f2px_, batch_first=True, padding_value=-1)

        # torch.cuda.synchronize()
        # new_end_time = time.time()
        # print(
        #     f"\tf2px Time: {(new_end_time - new_start_time):.4f} seconds")

        # torch.cuda.synchronize()
        # new_start_time = time.time()

        f2px_dummy = torch.vstack(
            [-1 * torch.ones_like(f2px_[0, :]).unsqueeze(0), f2px_]).to(mesh.device).to(torch.float32)
        f2px_dummy[f2px_dummy < 0] = torch.nan
        px_all = torch.sort(f2px_dummy[mesh.f_k_ring + 1].reshape(
            mesh.f_k_ring.shape[0], -1), dim=1).values
        cut_px = (px_all >= 0).int().sum(dim=1).max()
        px_all = px_all[:, :cut_px]
        p_indices = torch.nan_to_num(px_all, nan=-1).to(mesh.f.dtype)

        # torch.cuda.synchronize()
        # new_end_time = time.time()
        # print(
        #     f"\tNew p_indices Time: {(new_end_time - new_start_time):.4f} seconds")

        # Smax x |F|
        p_indices_T = p_indices.unsqueeze(-1)
        # p_indices = p_indices.squeeze(-1).T
        p_indices = p_indices.squeeze(-1).permute(1, 0)
        p_indices_extended = torch.clamp(p_indices, min=0)
        if not p_indices_extended.is_contiguous():
            p_indices_extended = p_indices_extended.contiguous()
        max_sample_size = p_indices.shape[0]
        # t_indices = (p_indices >= 0).any(dim=0).nonzero().squeeze()
        t_indices = (p_indices >= -1).any(dim=0).nonzero().squeeze()

        # Selected faces: mesh.f[t_indices]
        f_selected = torch.gather(
            mesh.f, 0, t_indices.unsqueeze(-1).expand(-1, mesh.f.shape[-1]))

    # Compute inside/outside and triangle distance for each sample-triangle pair
    # D_ij = sigmoid(δ_ij · d^2(i, j)/σ)
    # Smax x |F|
    inside_triangle = is_point_in_triangle_mat(
        samples, p_indices_extended, v, f_selected)
    squared_distances = point_triangle_squared_distance_mat(
        samples, p_indices_extended, v, f_selected)
    sign_squared_distances = (2 * inside_triangle - 1) * squared_distances
    p_dist_raw = sign_squared_distances / sigma
    p_sigmoid_raw = torch.sigmoid(p_dist_raw)

    # Set the values corresponding to invalid neighor triangles to 0
    p_sigmoid = torch.where(p_indices < 0, torch.tensor(0.), p_sigmoid_raw)

    if to_normalize:
        p_count = torch.where(p_indices < 0, torch.tensor(0.),
                              torch.ones_like(p_sigmoid))
    else:
        p_count = torch.zeros_like(p_sigmoid)

    with torch.no_grad():
        samples_pixel = torch.floor(samples).int()

        # Compute colors
        # |F| x 3
        # Note that we use the corresponding integer coordinates for the sample points (which are all 0.5-offset)
        image_array = np.array(image)
        if len(image_array.shape) == 2:
            image_array = image_array[..., np.newaxis]
        samples_pixel[:, 0] = torch.clamp(
            samples_pixel[:, 0], 0, image.width - 1)
        samples_pixel[:, 1] = torch.clamp(
            samples_pixel[:, 1], 0, image.height - 1)
        C = torch.from_numpy(
            image_array[samples_pixel[:, 1].cpu().int(), samples_pixel[:, 0].cpu().int()] / 255.0).type(v.dtype)
        C = C.to(mesh.device)

        # |F| x Smax x 3
        FC = torch.zeros(
            (mesh.f.shape[0], max_sample_size, C.shape[1]), dtype=v.dtype, device=mesh.device)
        # FC = C[p_indices_extended.T]
        p_indices_extended_t = p_indices_extended.permute(1, 0).long()
        if not p_indices_extended_t.is_contiguous():
            p_indices_extended_t = p_indices_extended_t.contiguous()
        FC = torch.gather(C, 0, p_indices_extended_t.view(-1, 1).long().expand(
            -1, C.shape[-1])).view(list(p_indices_extended_t.shape)+[C.shape[-1]])

        # Smax x |F| x 3
        # FC = FC.permute(1, 0, 2)

        # |F| x Smax
        face_mask_tensor = torch.tensor(face_mask, device=mesh.device).long()
        C_mean_gather = torch.gather(C_mean, 0,
                                     face_mask_tensor.unsqueeze(
                                         -1).expand(-1, C_mean.shape[-1])
                                     ).view([len(face_mask), C_mean.shape[-1]]).unsqueeze(1)
        # color_var = (FC - C_mean[face_mask, :].unsqueeze(1)) ** 2
        color_var = (FC - C_mean_gather) ** 2
        color_var = torch.where(p_indices_T < 0, torch.tensor(0.), color_var)
        # f_color_var = color_var.sum(axis=2).sum(
        #     axis=1) / (p_indices_T >= 0).int().squeeze(-1).sum(axis=1)
        color_var = color_var.sum(axis=2)

    # Soft color
    # Smax x |F| x 3
    # soft_color = (p_sigmoid.unsqueeze(-1) * FC).sum(dim=0)
    soft_color = p_sigmoid * color_var.T

    return soft_color, p_indices, p_sigmoid, p_count


def variance_loss(mesh: TriangleMesh, image,
                  samples, in_faces, sample_weights,
                  C_mean,
                  sigma, k_ring,
                  loss_mask=None, to_vis=False):
    face_mask = list(range(mesh.f.shape[0]))
    if isinstance(loss_mask, torch.Tensor):
        face_mask = loss_mask.nonzero().squeeze().cpu().numpy().tolist()

    f_color_var, p_indices, p_sigmoid, p_count = sample_softras(mesh, image, sigma, k_ring,
                                                                samples, in_faces, C_mean,
                                                                loss_mask, to_normalize=False)
    # f_color_var = f_color_var / torch.clamp(p_count.sum(dim=0), min=1)
    p_indices_extended = torch.clamp(p_indices, min=0)
    sample_weights_extended = torch.where(
        p_indices < 0, torch.tensor(0.), sample_weights[p_indices_extended])
    f_color_var = (f_color_var * sample_weights_extended).sum(dim=0)
    var_loss = f_color_var.mean()

    interior_samples = samples[in_faces < mesh.f_interior_count]
    if isinstance(loss_mask, torch.Tensor):
        f_color_var_ret = torch.zeros(
            mesh.f.shape[0], dtype=mesh.v.dtype, device=mesh.device)
        f_color_var_ret[face_mask] = f_color_var
    else:
        f_color_var_ret = f_color_var

    return var_loss, (samples, p_indices, p_sigmoid, interior_samples), C_mean, f_color_var_ret


@torch.no_grad()
def color_variance(v, f, samples_, in_faces, sample_weights,
                   image, face_mask=None, is_barycentric=False, f2px=None):
    if is_barycentric:
        samples = samples_[:, 0].unsqueeze(-1) * v[f[in_faces][:, 0], :] + \
            samples_[:, 1].unsqueeze(-1) * v[f[in_faces][:, 1], :] + \
            samples_[:, 2].unsqueeze(-1) * v[f[in_faces][:, 2], :]
    else:
        samples = samples_

    def is_iterable(obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    start_time = time.time()

    if not is_iterable(face_mask):
        face_mask = list(range(f.shape[0]))
    else:
        face_mask = sorted(face_mask)
    face_mask_set = set(face_mask)

    # Sample within the vector object boundary
    # |S| x 3
    with torch.no_grad():
        if samples.shape[0] < 2:
            return None, None

        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).to(v.device)
        if isinstance(in_faces, np.ndarray):
            in_faces = torch.from_numpy(in_faces).to(v.device)
        samples_pixel = torch.floor(samples).int()

        # Face to pixels
        if isinstance(f2px, torch.Tensor):
            f2px_ = f2px
        else:
            px_positions = torch.arange(in_faces.shape[0], device=v.device)
            sorted_vals, sorted_indices = torch.sort(in_faces)
            sorted_px_positions = px_positions[sorted_indices]
            counts = torch.bincount(in_faces)
            f2px_ = list(torch.split(sorted_px_positions, counts.tolist()))
            for i in range(f.shape[0] - len(f2px_)):
                f2px_.append(torch.tensor([], device=v.device))
            f2px_ = torch.nn.utils.rnn.pad_sequence(
                f2px_, batch_first=True, padding_value=-1)

        f2px_dummy = torch.vstack(
            [-1 * torch.ones_like(f2px_[0, :]).unsqueeze(0), f2px_]).to(v.device).to(torch.float32)
        f2px_dummy[f2px_dummy < 0] = torch.nan
        f_k_ring2 = torch.tensor(face_mask, device=v.device)
        px_all = torch.sort(f2px_dummy[f_k_ring2 + 1].reshape(
            len(face_mask), -1), dim=1).values
        cut_px = (px_all >= 0).int().sum(dim=1).max()
        px_all = px_all[:, :cut_px]
        p_indices = torch.nan_to_num(px_all, nan=-1).to(f.dtype)

        # Smax x |F|
        p_indices_T = p_indices.unsqueeze(-1)
        p_indices = p_indices.squeeze(-1).T
        p_indices_extended = torch.clamp(p_indices, min=0)
        max_sample_size = p_indices.shape[0]

    with torch.no_grad():
        # Compute colors
        # |F| x 3
        # Note that we use the corresponding integer coordinates for the sample points (which are all 0.5-offset)
        image_array = np.array(image)
        if len(image_array.shape) == 2:
            image_array = image_array[..., np.newaxis]
        samples_pixel[:, 0] = torch.clamp(
            samples_pixel[:, 0], 0, image.width - 1)
        samples_pixel[:, 1] = torch.clamp(
            samples_pixel[:, 1], 0, image.height - 1)
        C = torch.from_numpy(
            image_array[samples_pixel[:, 1].cpu().int(), samples_pixel[:, 0].cpu().int()] / 255.0).type(v.dtype)
        C = C.to(v.device)

        C_mean = torch.zeros(
            (f.shape[0], C.shape[-1]), dtype=v.dtype, device=v.device)
        for d in range(C.shape[-1]):
            scatter_mean(C[:, d], in_faces.long(), out=C_mean[:, d])

        # scatter_mean(C[:, 0], in_faces.long(), out=C_mean[:, 0])
        # scatter_mean(C[:, 1], in_faces.long(), out=C_mean[:, 1])
        # scatter_mean(C[:, 2], in_faces.long(), out=C_mean[:, 2])

        # |F| x Smax x 3
        FC = torch.zeros(
            (f.shape[0], max_sample_size, C.shape[1]), dtype=v.dtype, device=v.device)
        FC = C[p_indices_extended.T]

        # |F| x Smax
        color_var = (FC - C_mean[face_mask, :].unsqueeze(1)) ** 2
        color_var = torch.where(p_indices_T < 0, torch.tensor(0.), color_var)

        sample_weights_extended = torch.where(
            p_indices < 0, torch.tensor(0.), sample_weights[p_indices_extended])
        color_var = color_var * \
            sample_weights_extended.permute(1, 0).unsqueeze(-1)

        f_color_var = color_var.sum(axis=2).sum(axis=1)

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"color variance all triangles Time: {execution_time:.4f} seconds")

    f_color_var_ret = f_color_var

    return C_mean, f_color_var_ret


@torch.no_grad()
def color_variance_sampling(v, f, image, step_size=0.25, min_samples=10):
    face_mask = list(range(f.shape[0]))
    samples, in_faces = sample_triangle_stratified(
        v, f, face_mask, step_size=step_size, min_samples=min_samples)
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).to(v.device)
    if isinstance(in_faces, np.ndarray):
        in_faces = torch.from_numpy(in_faces).to(v.device)

    sample_weights = compute_sample_weights(v, f, in_faces)

    C_mean, f_color_var_ret = color_variance(v, f, samples, in_faces,
                                             sample_weights, image, face_mask=None)

    return C_mean, f_color_var_ret
