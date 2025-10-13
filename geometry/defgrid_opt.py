import time

import igl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_scatter import scatter_mean

from geometry.mesh_triangle import TriangleMesh
from geometry.defgrid import find_k_ring, is_point_in_triangle_mat, point_triangle_distance_mat
from tools.plot_utils import plot_association_func, plot_mesh
from tools.param import epsilon


def variance_loss(mesh: TriangleMesh, vertices,
                  image, sigma, f_weights=[], k_ring=3, euclidean_threshold=-1, to_vis=False):
    start_time = time.time()

    # Find the K-ring of the triangle for efficient computation
    # We assume a triangle is only to be deformed slightly
    # so the variance computation can be local
    if len(mesh.tt) == 0:
        with torch.no_grad():
            mesh.tt, _ = igl.triangle_triangle_adjacency(mesh.f.cpu().numpy())

    # The actual positions
    v = vertices  # TODO: Check this line
    mesh_sample_offset = torch.tensor(
        [0.5, 0.5, 0], dtype=v.dtype, device='cpu')

    # Precompute an aabb
    if len(mesh.cache_p_min) == 0:
        p_min, p_max = mesh.aabb(pixel_aligned=True)
        p_min = torch.floor(p_min)
        p_max = torch.ceil(p_max)
        padding = 10
        if euclidean_threshold >= 0:
            padding = 0
        mesh.cache_p_min = (p_min - torch.tensor([padding, padding, 0],
                                                 dtype=p_min.dtype, device=p_min.device)).cpu()
        mesh.cache_p_max = (p_max + torch.tensor([padding, padding, 0],
                                                 dtype=p_min.dtype, device=p_min.device)).cpu()

    # Sample within the vector object boundary
    # |S| x 3
    with torch.no_grad():
        sample_start_time = time.time()
        samples, in_faces = mesh.grid_sample_triangle_aabb(
            list(range(mesh.f.shape[0])),
            mesh.cache_p_min + mesh_sample_offset, mesh.cache_p_max + mesh_sample_offset,
            step_size=1)
        end_time = time.time()
        # samples = torch.from_numpy(samples).type(v.dtype).to(mesh.device)
        # in_faces = torch.from_numpy(in_faces).int().to(mesh.device)
        samples = torch.from_numpy(samples).to(mesh.device)
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
            k_f_start_time = time.time()

            # Use the k-ring as our domain
            f_k_ring = find_k_ring(mesh.tt, fid, k_ring)
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
            # if euclidean_threshold >= 0 and fid in mesh.boundary_fid:
            #     sample_indices = filter_samples_euclidean(
            #         samples, sample_indices, v[mesh.f[fid]], euclidean_threshold)
            max_sample_size = max(max_sample_size, sample_indices.shape[0])

            sample_lookup_end_time = time.time()
            sample_lookup_time += sample_lookup_end_time - sample_lookup_start_time

            p_indices.append(sample_indices.int().reshape(-1, 1))
            t_indices.append(int(fid))
            f_k_rings.append(int_f_k_ring)

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
        p_indices = p_indices.squeeze(-1).T
        p_indices_extended = torch.clamp(p_indices, min=0)

        # Smax x |F| x Kmax
        neighborhoods = -1 * \
            torch.ones(
                max_sample_size, mesh.f_interior_count, max_neighborhood_size, dtype=f_k_rings[0].dtype, device=mesh.device)
        f_k_rings = torch.nn.utils.rnn.pad_sequence(
            f_k_rings, batch_first=True, padding_value=-1)
        neighborhoods[:, t_indices, :] = f_k_rings.squeeze(-1)
        neighborhoods_extended = torch.clamp(neighborhoods, min=0)

        # Find the corresponding vertices for the triangle neighborhoods
        # Smax x |F| x Kmax
        neighborhood_vertices = mesh.f[neighborhoods_extended]

    # Compute inside/outside and triangle distance for each sample-triangle pair
    # Smax x |F| x Kmax
    inside_triangle = is_point_in_triangle_mat(
        samples, p_indices_extended, neighborhood_vertices, v)
    distances = point_triangle_distance_mat(
        samples, p_indices_extended, neighborhood_vertices, v)
    sign_distances = (2 * inside_triangle - 1) * distances
    p_dist_raw = sign_distances / sigma

    # Set the values corresponding to invalid neighor triangles to 0
    p_dist_raw2 = torch.where(neighborhoods < 0, torch.tensor(0.), p_dist_raw)
    p_dist = torch.where(p_indices.unsqueeze(-1) < 0,
                         torch.tensor(0.), p_dist_raw2)

    # Shift the distance for more stable softmax
    p_dist = p_dist - p_dist.max(dim=2).values.unsqueeze(-1)
    p_dist = torch.exp(p_dist)

    p_dist_sum = p_dist.sum(dim=2)

    # Set the values corresponding to invalid neighor triangles to 0
    p_dist_sum = torch.where(p_dist_sum == 0, torch.tensor(1.), p_dist_sum)
    p_dist_sum = torch.where(p_dist_sum < epsilon,
                             torch.tensor(epsilon), p_dist_sum)

    # Note that we ensure the first element of the k-ring list is the triangle itself
    p_fk_dist = p_dist[:, :, 0]
    p_ff_asso = p_fk_dist / p_dist_sum

    p_count = torch.where(p_indices < 0, torch.tensor(0.),
                          torch.ones_like(p_ff_asso))

    with torch.no_grad():
        # Compute colors
        # |F| x 3
        # Note that we use the corresponding integer coordinates for the sample points (which are all 0.5-offset)
        image_array = np.array(image)
        if len(image_array.shape) == 2:
            image_array = image_array[..., np.newaxis]
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
        color_var = (FC - C_mean.unsqueeze(1)) ** 2
        # color_var[p_indices.T < 0] = 0
        color_var = torch.where(p_indices_T < 0, torch.tensor(0.), color_var)
        color_var = color_var.sum(axis=2)

    # Weight the color variance and average
    # Smax x |F|
    if len(f_weights) == 0:
        var_loss = ((p_ff_asso * color_var.T).sum(dim=0) /
                    torch.clamp(p_count.sum(dim=0), min=1)).mean()
        with torch.no_grad():
            f_color_var = (p_ff_asso * color_var.T).sum(dim=0) / \
                torch.clamp(p_count.sum(dim=0), min=1)
    else:
        var_loss = ((p_ff_asso * color_var.T).sum(dim=0) /
                    torch.clamp(p_count.sum(dim=0), min=1) * f_weights).mean()
        with torch.no_grad():
            f_color_var = (p_ff_asso * color_var.T).sum(dim=0) / \
                torch.clamp(p_count.sum(dim=0), min=1) * f_weights

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"variance_loss all triangles Time: {execution_time:.4f} seconds")

    # Visualize the association function for debugging
    with torch.no_grad():
        if to_vis:
            fid = 556
            ff_debug = mesh.f[fid]
            print(v[ff_debug])
            ax = plot_mesh(mesh)

            p_ff_asso_debug = torch.zeros(
                samples.shape[0], 1, dtype=v.dtype, device=mesh.device)
            p_indices_face = p_indices[:, fid]
            p_ff_asso_face = p_ff_asso[:, fid]
            p_indices_face_valid = p_indices_face[p_indices_face >= 0]
            p_ff_asso_debug[p_indices_face_valid] = p_ff_asso_face[p_indices_face >=
                                                                   0].view(-1, 1)
            p_ff_asso_debug = p_ff_asso_debug.squeeze()

            plot_association_func(
                samples, p_ff_asso_debug, mesh, zoom=1.0, ax=ax)
            # plt.show(block=True)
            # plt.show()
            plt.savefig(f'./asso_func.png', dpi=600)
            # exit()

    interior_samples = samples[in_faces < mesh.f_interior_count]

    return var_loss, (samples, p_indices, p_ff_asso, interior_samples), C_mean, f_color_var
