import time

import igl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_scatter import scatter_mean

from geometry.defgrid import find_k_ring_vf
from geometry.mesh_triangle import TriangleMesh
from largesteps.parameterize import to_differential
from tools.plot_utils import plot_association_func, plot_mesh, to_pil_image
from tools.param import epsilon
from learning.sampler import (
    grid_sample_triangle, grid_sample_triangle_aabb, sample_triangle_stratified)


def is_point_in_triangle_mat(samples, p_indices_extended, v, f):
    # Calculate barycentric coordinates
    # V0 = v[f[:, 0]].unsqueeze(0)
    # V1 = v[f[:, 1]].unsqueeze(0)
    # V2 = v[f[:, 2]].unsqueeze(0)

    VV = torch.gather(v, 0, f.view(-1, 1).long().expand(-1,
                      v.shape[-1])).reshape(-1, f.shape[-1], v.shape[-1])
    V0 = VV[:, 0, :].unsqueeze(0)
    V1 = VV[:, 1, :].unsqueeze(0)
    V2 = VV[:, 2, :].unsqueeze(0)

    detT = (V1[:, :, 1] - V2[:, :, 1]) * (V0[:, :, 0] - V2[:, :, 0]) + \
        (V2[:, :, 0] - V1[:, :, 0]) * (V0[:, :, 1] - V2[:, :, 1])

    # samples_x = samples[:, 0][p_indices_extended]
    # samples_y = samples[:, 1][p_indices_extended]
    samples_picked = torch.gather(samples, 0, p_indices_extended.view(-1, 1).long().expand(
        -1, samples.shape[-1])).view(list(p_indices_extended.shape)+[samples.shape[-1]])
    samples_x = samples_picked[:, :, 0]
    samples_y = samples_picked[:, :, 1]

    alpha = ((V1[:, :, 1] - V2[:, :, 1]) * (samples_x - V2[:, :, 0]) +
             (V2[:, :, 0] - V1[:, :, 0]) * (samples_y - V2[:, :, 1])) / detT
    beta = ((V2[:, :, 1] - V0[:, :, 1]) * (samples_x - V2[:, :, 0]) +
            (V0[:, :, 0] - V2[:, :, 0]) * (samples_y - V2[:, :, 1])) / detT
    gamma = 1 - alpha - beta

    # Check if points are inside the triangle
    inside_triangle = torch.where((alpha >= 0) & (
        beta >= 0) & (gamma >= 0), 1, 0).to(v.dtype)

    return inside_triangle


def point_triangle_squared_distance_mat(samples, p_indices_extended, v, f):
    def point_edge_distance(points, v0, v1):
        line_vector = (v1 - v0)
        point_vector = points - v0
        projection = torch.sum(point_vector * line_vector, dim=-1) / \
            torch.sum(line_vector * line_vector, dim=-1)
        projection = torch.clamp(projection, 0, 1)
        closest_point = v0 + projection.unsqueeze(-1) * line_vector

        # Small value to ensure numerical stability
        # distance = torch.sqrt(
        #     torch.sum((points - closest_point) ** 2, dim=-1) + epsilon)
        # Soft rasterizer paper uses d^2
        distance = torch.sum((points - closest_point) ** 2, dim=-1)

        return distance

    # V0 = v[f[:, 0]].unsqueeze(0)
    # V1 = v[f[:, 1]].unsqueeze(0)
    # V2 = v[f[:, 2]].unsqueeze(0)

    VV = torch.gather(v, 0, f.view(-1, 1).long().expand(-1,
                      v.shape[-1])).reshape(-1, f.shape[-1], v.shape[-1])
    # assert torch.all(VV == torch.stack(
    #     [v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]], dim=1))
    V0 = VV[:, 0, :].unsqueeze(0)
    V1 = VV[:, 1, :].unsqueeze(0)
    V2 = VV[:, 2, :].unsqueeze(0)

    # samples_picked = samples[p_indices_extended]
    samples_picked = torch.gather(samples, 0, p_indices_extended.view(-1, 1).long().expand(
        -1, samples.shape[-1])).view(list(p_indices_extended.shape)+[samples.shape[-1]])
    # assert torch.all(samples_picked == samples_picked_gather)

    distances = torch.stack([point_edge_distance(samples_picked, V0, V1),
                             point_edge_distance(samples_picked, V1, V2),
                             point_edge_distance(samples_picked, V2, V0)])
    distances = torch.min(distances, dim=0).values

    return distances.to(v.dtype)


def variance_loss(mesh: TriangleMesh, image, sigma, k_ring, f_weights=[],
                  to_normalize=True, loss_mask=None, to_vis=False):
    start_time = time.time()

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

    # The actual positions
    v = mesh.get_v()
    mesh_sample_offset = torch.tensor(
        [0.5, 0.5, 0], dtype=v.dtype, device='cpu')

    # Precompute an aabb
    if len(mesh.cache_p_min) == 0:
        p_min, p_max = mesh.aabb(pixel_aligned=True)
        p_min = torch.floor(p_min)
        p_max = torch.ceil(p_max)
        padding = 10
        mesh.cache_p_min = (p_min - torch.tensor([padding, padding, 0],
                                                 dtype=p_min.dtype, device=p_min.device)).cpu()
        mesh.cache_p_max = (p_max + torch.tensor([padding, padding, 0],
                                                 dtype=p_min.dtype, device=p_min.device)).cpu()

    # Sample within the vector object boundary
    # |S| x 3
    with torch.no_grad():
        sample_start_time = time.time()
        face_mask = list(range(mesh.f.shape[0]))
        if isinstance(loss_mask, torch.Tensor):
            face_mask = loss_mask.nonzero().squeeze().cpu().numpy().tolist()
        face_mask_set = set(face_mask)

        # samples, in_faces = sample_triangle_stratified(
        #     v, mesh.f, face_mask, step_size=2.5, min_samples=4)
        samples, in_faces = grid_sample_triangle_aabb(v, mesh.f,
                                                      face_mask,
                                                      mesh.cache_p_min + mesh_sample_offset, mesh.cache_p_max + mesh_sample_offset,
                                                      c_offset=-len(mesh.attributes), step_size=1)
        print(samples.shape[0])
        end_time = time.time()
        # samples = torch.from_numpy(samples).type(v.dtype).to(mesh.device)
        # in_faces = torch.from_numpy(in_faces).int().to(mesh.device)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).to(mesh.device)
        if isinstance(in_faces, np.ndarray):
            in_faces = torch.from_numpy(in_faces).to(mesh.device)
        samples_pixel = torch.floor(samples).int()
        sample_end_time = time.time()
        # print(
        #     f"\t\tgrid_sample_triangle Time: {(end_time - sample_start_time):.4f} seconds")
        # print(
        #     f"\t\tAfter grid_sample_triangle Time: {(sample_end_time - end_time):.4f} seconds")
        # print(
        #     f"\tSample Time: {(sample_end_time - sample_start_time):.4f} seconds")

        # Fetch the neighborhood per triangle per sample
        # Use the largest neighborhood size to make a dense but efficient matrix
        max_neighborhood_size = 0
        max_sample_size = 0
        p_indices = []
        t_indices = []
        f_k_rings = []

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
            t_indices.append(int(fid))

            save_time += time.time() - sample_lookup_end_time

        kring_end_time = time.time()
        kring_execution_time = kring_end_time - kring_start_time
        # print(
        #     f"\tvariance_loss k-ring Time: {kring_execution_time:.4f} seconds")
        # print(f"\t\tvariance_loss k-ring Time: {k_ring_time:.4f} seconds")
        # print(
        #     f"\t\tvariance_loss sample lookup Time: {sample_lookup_time:.4f} seconds")
        # print(
        #     f"\t\tvariance_loss save Time: {save_time:.4f} seconds")

        p_indices = torch.nn.utils.rnn.pad_sequence(
            p_indices, batch_first=True, padding_value=-1)
        # Smax x |F|
        p_indices_T = p_indices
        p_indices = p_indices.squeeze(-1).permute(1, 0)
        p_indices_extended = torch.clamp(p_indices, min=0)

    # Compute inside/outside and triangle distance for each sample-triangle pair
    # D_ij = sigmoid(δ_ij · d^2(i, j)/σ)
    # Smax x |F|
    inside_triangle = is_point_in_triangle_mat(
        samples, p_indices_extended, v, mesh.f[t_indices])
    squared_distances = point_triangle_squared_distance_mat(
        samples, p_indices_extended, v, mesh.f[t_indices])
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

        C_mean_all = torch.zeros(
            (mesh.f.shape[0], C.shape[-1]), dtype=v.dtype, device=mesh.device)
        for d in range(C.shape[-1]):
            scatter_mean(C[:, d], in_faces.long(), out=C_mean_all[:, d])

        # scatter_mean(C[:, 0], in_faces.long(), out=C_mean_all[:, 0])
        # scatter_mean(C[:, 1], in_faces.long(), out=C_mean_all[:, 1])
        # scatter_mean(C[:, 2], in_faces.long(), out=C_mean_all[:, 2])
        C_mean = C_mean_all[0:mesh.f_interior_count, :]

        # |F| x Smax x 3
        FC = torch.zeros(
            (mesh.f.shape[0], max_sample_size, C.shape[1]), dtype=v.dtype, device=mesh.device)
        FC = C[p_indices_extended.T]

        # |F| x Smax
        color_var = (FC - C_mean[face_mask, :].unsqueeze(1)) ** 2
        # color_var[p_indices.T < 0] = 0
        color_var = torch.where(p_indices_T < 0, torch.tensor(0.), color_var)
        color_var = color_var.sum(axis=2)

    # Weight the color variance and average
    # Smax x |F|
    if len(f_weights) == 0:
        f_color_var = (p_sigmoid * color_var.T).sum(dim=0) / \
            torch.clamp(p_count.sum(dim=0), min=1)
        var_loss = f_color_var.mean()
    else:
        f_color_var = (p_sigmoid * color_var.T).sum(dim=0) / \
            torch.clamp(p_count.sum(dim=0), min=1) * f_weights
        var_loss = (f_color_var).mean()

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"variance_loss all triangles Time: {execution_time:.4f} seconds")

    # Visualize the association function for debugging
    with torch.no_grad():
        if to_vis:
            fid = 5
            ff_debug = mesh.f[fid]
            print(v[ff_debug])
            ax = plot_mesh(mesh)

            p_sigmoid_debug = torch.zeros(
                samples.shape[0], 1, dtype=v.dtype, device=mesh.device)
            p_indices_face = p_indices[:, fid]
            p_sigmoid_face = p_sigmoid[:, fid]
            p_indices_face_valid = p_indices_face[p_indices_face >= 0]
            p_sigmoid_debug[p_indices_face_valid] = p_sigmoid_face[p_indices_face >=
                                                                   0].view(-1, 1)
            p_sigmoid_debug = p_sigmoid_debug.squeeze()

            ax = plot_association_func(
                samples, p_sigmoid_debug, mesh, zoom=1.0, ax=ax)
            ax.set_title(f'SoftRas: Triangle Occupancy f: {fid}')

            # plt.show(block=True)
            # plt.show()
            plt.savefig(f'./asso_func_softras.png', dpi=600)
            # exit()

    interior_samples = samples[in_faces < mesh.f_interior_count]
    if isinstance(loss_mask, torch.Tensor):
        f_color_var_ret = torch.zeros(
            mesh.f.shape[0], dtype=v.dtype, device=mesh.device)
        f_color_var_ret[face_mask] = f_color_var
    else:
        f_color_var_ret = f_color_var

    return var_loss, (samples, p_indices, p_sigmoid, interior_samples), C_mean, f_color_var_ret


@torch.no_grad()
def color_variance_px(v, f, image, face_mask=None):
    start_time = time.time()

    def is_iterable(obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    if not is_iterable(face_mask):
        face_mask = list(range(f.shape[0]))
    else:
        face_mask = sorted(face_mask)
    face_mask_set = set(face_mask)

    # The actual positions
    mesh_sample_offset = torch.tensor(
        [0.5, 0.5, 0], dtype=v.dtype, device='cpu')

    # Precompute an aabb
    ff = f[face_mask]
    ff = ff.flatten()
    p_min, p_max = v[ff, :].min(
        axis=0).values, v[ff, :].max(axis=0).values
    p_min = torch.floor(p_min)
    p_max = torch.ceil(p_max)

    cache_p_min = p_min.cpu()
    cache_p_max = p_max.cpu()

    # Sample within the vector object boundary
    # |S| x 3
    with torch.no_grad():
        sample_start_time = time.time()

        samples, in_faces = grid_sample_triangle_aabb(v, f,
                                                      face_mask,
                                                      cache_p_min + mesh_sample_offset, cache_p_max + mesh_sample_offset,
                                                      c_offset=-10, step_size=1)

        # print(samples.shape[0])
        end_time = time.time()

        if samples.shape[0] < 2:
            return None, None

        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).to(v.device)
        if isinstance(in_faces, np.ndarray):
            in_faces = torch.from_numpy(in_faces).to(v.device)
        samples_pixel = torch.floor(samples).int()
        sample_end_time = time.time()
        # print(
        #     f"\t\tgrid_sample_triangle Time: {(end_time - sample_start_time):.4f} seconds")
        # print(
        #     f"\t\tAfter grid_sample_triangle Time: {(sample_end_time - end_time):.4f} seconds")
        # print(
        #     f"\tSample Time: {(sample_end_time - sample_start_time):.4f} seconds")

        # Fetch the neighborhood per triangle per sample
        # Use the largest neighborhood size to make a dense but efficient matrix
        max_neighborhood_size = 0
        max_sample_size = 0
        p_indices = []
        t_indices = []
        f_k_rings = []

        k_ring_time = 0
        sample_lookup_time = 0
        save_time = 0
        kring_start_time = time.time()
        for fid in range(f.shape[0]):
            if fid not in face_mask_set:
                continue

            f_k_ring = [fid]

            f_k_ring = torch.tensor(f_k_ring)
            f_k_ring = f_k_ring.unsqueeze(1)
            max_neighborhood_size = max(
                max_neighborhood_size, f_k_ring.shape[0])
            f_k_ring = f_k_ring.to(v.device)

            if f_k_ring.shape[0] == 0:
                continue

            int_f_k_ring = f_k_ring
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

        p_indices = torch.nn.utils.rnn.pad_sequence(
            p_indices, batch_first=True, padding_value=-1)
        # Smax x |F|
        p_indices_T = p_indices
        p_indices = p_indices.squeeze(-1).T
        p_indices_extended = torch.clamp(p_indices, min=0)

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
        # f_color_var = color_var.sum(axis=2).sum(
        #     axis=1) / (p_indices_T >= 0).int().squeeze(-1).sum(axis=1)
        f_color_var = color_var.sum(axis=2).sum(axis=1)

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"color variance all triangles Time: {execution_time:.4f} seconds")

    f_color_var_ret = f_color_var

    return C_mean, f_color_var_ret
