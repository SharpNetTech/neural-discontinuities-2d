import time

import igl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_scatter import scatter_mean

from geometry.mesh_triangle import TriangleMesh
from largesteps.parameterize import to_differential
from tools.plot_utils import plot_association_func, plot_mesh, to_pil_image
from tools.utils import get_color_at_position
from tools.math_utils import slice_torch_sparse_coo_tensor
from tools.param import epsilon
from learning.sampler import grid_sample_triangle, grid_sample_triangle_aabb


def find_k_ring(tt, fid, k):
    # If k is negative, return all triangles
    if k < 0:
        return [fid] + list([tid for tid in range(tt.shape[0]) if tid != fid])

    # Find the K-ring of the triangle
    # We make sure the first element is the triangle itself
    f_k_ring = [fid]
    for _ in range(k):
        f_k_ring = list(set(f_k_ring + tt[f_k_ring].ravel().tolist()))
        f_k_ring = [fid for fid in f_k_ring if fid > -1]
    f_k_ring.remove(fid)
    f_k_ring = [fid] + f_k_ring

    return f_k_ring


def find_k_ring_vf(vf, f, fid, k):
    # If k is negative, return all triangles
    if k < 0:
        return [fid] + list([tid for tid in range(f.shape[0]) if tid != fid])

    # Find the K-ring of the triangle using v-f adjacency
    # We make sure the first element is the triangle itself
    f_k_ring = [fid]
    for _ in range(k):
        vid = f[f_k_ring]
        f_k_ring = list(set(f_k_ring + vf[vid].ravel().tolist()))
        f_k_ring = [fid for fid in f_k_ring if fid > -1]
    f_k_ring.remove(fid)
    f_k_ring = [fid] + f_k_ring

    return f_k_ring


def is_point_in_triangle(points, vertices):
    """
    Check if points are inside a triangle.

    Parameters:
    - points (torch.Tensor): Coordinates of points (Nx2).
    - vertices (torch.Tensor): Coordinates of the triangle vertices (3x2).

    Returns:
    - torch.Tensor: Binary tensor indicating if each point is inside the triangle.
    """

    # Calculate barycentric coordinates
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    detT = (v1[1] - v2[1]) * (v0[0] - v2[0]) + \
        (v2[0] - v1[0]) * (v0[1] - v2[1])

    alpha = ((v1[1] - v2[1]) * (points[:, 0] - v2[0]) +
             (v2[0] - v1[0]) * (points[:, 1] - v2[1])) / detT
    beta = ((v2[1] - v0[1]) * (points[:, 0] - v2[0]) +
            (v0[0] - v2[0]) * (points[:, 1] - v2[1])) / detT
    gamma = 1 - alpha - beta

    # Check if points are inside the triangle
    inside_triangle = torch.zeros(
        points.shape[0], dtype=torch.float32, device=points.device)
    inside_triangle += torch.where((alpha >= 0)
                                   & (beta >= 0) & (gamma >= 0), 1, 0)

    return inside_triangle


def is_point_in_triangle_mat(samples, p_indices_extended, neighborhood_vertices, v):
    # Calculate barycentric coordinates
    V0 = v[neighborhood_vertices[:, :, :, 0]]
    V1 = v[neighborhood_vertices[:, :, :, 1]]
    V2 = v[neighborhood_vertices[:, :, :, 2]]
    detT = (V1[:, :, :, 1] - V2[:, :, :, 1]) * (V0[:, :, :, 0] - V2[:, :, :, 0]) + \
        (V2[:, :, :, 0] - V1[:, :, :, 0]) * (V0[:, :, :, 1] - V2[:, :, :, 1])

    samples_x = samples[:, 0][p_indices_extended].unsqueeze(2)
    samples_y = samples[:, 1][p_indices_extended].unsqueeze(2)
    alpha = ((V1[:, :, :, 1] - V2[:, :, :, 1]) * (samples_x - V2[:, :, :, 0]) +
             (V2[:, :, :, 0] - V1[:, :, :, 0]) * (samples_y - V2[:, :, :, 1])) / detT
    beta = ((V2[:, :, :, 1] - V0[:, :, :, 1]) * (samples_x - V2[:, :, :, 0]) +
            (V0[:, :, :, 0] - V2[:, :, :, 0]) * (samples_y - V2[:, :, :, 1])) / detT
    gamma = 1 - alpha - beta

    # Check if points are inside the triangle
    inside_triangle = torch.where((alpha >= 0) & (
        beta >= 0) & (gamma >= 0), 1, 0).to(v.dtype)

    return inside_triangle


def point_triangle_distance(points, vertices):
    def point_edge_distance(points, v0, v1):
        line_vector = (v1 - v0).reshape(-1, 1)
        point_vector = points - v0
        projection = (point_vector @ line_vector) / \
            (line_vector.T @ line_vector)
        projection = torch.clamp(projection, 0, 1)
        closest_point = v0 + projection * line_vector.T

        distance = torch.sqrt(torch.sum((points - closest_point) ** 2, dim=1))

        return distance

    return torch.min(torch.vstack([point_edge_distance(points, vertices[0], vertices[1]),
                                  point_edge_distance(
                                      points, vertices[1], vertices[2]),
                                  point_edge_distance(points, vertices[2], vertices[0])]), dim=0).values


def filter_samples_euclidean(points, sample_indices, vertices, threshold):
    inside = is_point_in_triangle(points[sample_indices], vertices)
    inside = inside.bool()
    sample_indices2 = sample_indices[~inside]
    distances = point_triangle_distance(points[sample_indices2], vertices)

    return torch.cat((sample_indices2[distances < threshold], sample_indices[inside]), dim=0)


def point_triangle_distance_mat(samples, p_indices_extended, neighborhood_vertices, v):
    def point_edge_distance(points, v0, v1):
        line_vector = (v1 - v0)
        point_vector = points - v0
        projection = torch.sum(point_vector * line_vector, dim=-1) / \
            torch.sum(line_vector * line_vector, dim=-1)
        projection = torch.clamp(projection, 0, 1)
        closest_point = v0 + projection.unsqueeze(-1) * line_vector

        # Small value to ensure numerical stability
        distance = torch.sqrt(
            torch.sum((points - closest_point) ** 2, dim=-1) + epsilon)

        return distance

    V0 = v[neighborhood_vertices[:, :, :, 0]]
    V1 = v[neighborhood_vertices[:, :, :, 1]]
    V2 = v[neighborhood_vertices[:, :, :, 2]]

    samples_picked = samples[p_indices_extended].unsqueeze(2)
    samples_picked = samples[p_indices_extended].unsqueeze(2)

    distances = torch.stack([point_edge_distance(samples_picked, V0, V1),
                             point_edge_distance(samples_picked, V1, V2),
                             point_edge_distance(samples_picked, V2, V0)])
    distances = torch.min(distances, dim=0).values

    return distances.to(v.dtype)


def variance_loss_for(mesh: TriangleMesh, image, sigma, k_ring=3):
    start_time = time.time()

    # Find the K-ring of the triangle for efficient computation
    # We assume a triangle is only to be deformed slightly
    # so the variance computation can be local
    tt, _ = igl.triangle_triangle_adjacency(mesh.f.numpy())
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

    # Compute the color variance per triangle
    var_loss = 0
    # This can be a sparse matrix
    p_tri_association = torch.zeros(
        int(mesh.size[1]), int(mesh.size[0]), mesh.f_interior_count, dtype=v.dtype)
    C_mean = torch.zeros(mesh.f_interior_count, 3, dtype=v.dtype)
    for fid, ff in enumerate(mesh.f):
        # Use the K-ring as our domain
        # f_k_ring = find_k_ring(tt, fid, k_ring)
        f_k_ring = find_k_ring_vf(mesh.vf, mesh.f, fid, k_ring)

        # Sample the K-ring
        # For now, sample per pixel
        # (1-pixel step between two samples, coordinates are not exact)
        samples = []
        CC_mean = np.zeros((1, 3))
        for fid_k in f_k_ring:
            ss, _ = grid_sample_triangle(
                mesh.v, mesh.f, fid_k, c_offset=-len(mesh.attributes), step_size=1, pixel_aligned=True)
            if ss.shape[0] == 0:
                continue
            if fid_k == fid:
                C = np.array([get_color_at_position(image, xy[0], xy[1])
                              for xy in ss]) / 255.0
                CC_mean = C.mean(axis=0)
                C_mean[fid] = torch.from_numpy(CC_mean)
            samples.append(ss)
        samples = np.vstack(samples)
        C = np.array([get_color_at_position(image, xy[0], xy[1])
                      for xy in samples]) / 255.0
        color_var = ((C - CC_mean) ** 2).sum(axis=1)
        color_var = torch.tensor(color_var, dtype=v.dtype)
        samples = torch.tensor(samples, dtype=v.dtype)

        # Compute the soft pixel-triangle association function per pixel
        p_dist = torch.zeros(samples.shape[0], len(
            f_k_ring), dtype=v.dtype)
        ff_col = 0
        for i, fid_k in enumerate(f_k_ring):
            ff_k = mesh.f[fid_k]
            inside_triangle = is_point_in_triangle(samples, v[ff_k])
            distances = point_triangle_distance(samples, v[ff_k])
            sign_distances = (2 * inside_triangle - 1) * distances
            p_dist[:, i] = sign_distances
            if fid_k == fid:
                ff_col = i

        p_dist = p_dist / sigma
        p_dist = torch.exp(p_dist)
        p_ff_dist = p_dist[:, ff_col]
        p_ff_asso = p_ff_dist / p_dist.sum(dim=1)

        # Cache pixel-triangle association function
        p_tri_association[samples[:, 1].to(dtype=torch.int), samples[:, 0].to(
            dtype=torch.int), fid] = p_ff_asso

        # Weight the color variance
        color_var = (p_ff_asso * color_var).mean()

        var_loss += color_var

        # Calculate and print the execution time
        end_time = time.time()
        execution_time = end_time - start_time
        execution_time *= mesh.f_interior_count
        print(
            f"variance_loss_for estimated all triangles Time: {execution_time:.4f} seconds")

        # Visualize the association function for debugging
        ff_debug = mesh.f[fid]
        print(v[ff_debug])
        ax = plot_mesh(mesh)
        plot_association_func(samples, p_ff_asso, mesh, zoom=1.0, ax=ax)
        # plt.show(block=True)
        plt.show(block=False)

        break

    var_loss = var_loss / mesh.f_interior_count

    return var_loss, p_tri_association, C_mean


def variance_loss(mesh: TriangleMesh, image, sigma, f_weights=[], k_ring=3, euclidean_threshold=-1, to_vis=False):
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
        samples, in_faces = grid_sample_triangle_aabb(v, mesh.f,
                                                      list(
                                                          range(mesh.f.shape[0])),
                                                      mesh.cache_p_min + mesh_sample_offset, mesh.cache_p_max + mesh_sample_offset,
                                                      c_offset=-len(mesh.attributes), step_size=1)
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
            fid = 5
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

            ax = plot_association_func(
                samples, p_ff_asso_debug, mesh, zoom=1.0, ax=ax)
            ax.set_title(f'DefGrid: Triangle Occupancy f: {fid}')

            # plt.show(block=True)
            # plt.show()
            plt.savefig(f'./asso_func_defgrid.png', dpi=600)
            # exit()

    interior_samples = samples[in_faces < mesh.f_interior_count]

    return var_loss, (samples, p_indices, p_ff_asso, interior_samples), C_mean, f_color_var


def reconstruction_loss(image_norm, p_tri_association, C_mean, to_vis=False):
    object_mask = torch.zeros_like(image_norm, device=image_norm.device)

    if not isinstance(p_tri_association, tuple):
        f_hat = p_tri_association @ C_mean
        # reconst_image = to_pil_image(f_hat)
        # reconst_image.save(f'./reconst_image.png')
        assert False, 'No longer supported'
    else:
        samples, p_indices, p_ff_asso, obj_mask = p_tri_association

        p_tri_color = p_ff_asso.unsqueeze(-1) * C_mean.unsqueeze(0)
        mask = p_indices >= 0
        column_ids = torch.where(p_indices >= 0)[1]
        index_pairs = torch.stack((p_indices[mask], column_ids))
        sparse_p_tri_color = torch.sparse_coo_tensor(index_pairs, p_tri_color[mask].view(
            -1, p_tri_color.size(2)), torch.Size([samples.shape[0], C_mean.shape[0], p_tri_color.shape[-1]])).to(samples.dtype)
        sparse_p_tri_color = sparse_p_tri_color.coalesce()
        color_reconst = torch.sparse.sum(sparse_p_tri_color, dim=1).to_dense()

        # The input has white canvas
        f_hat = torch.ones_like(image_norm, device=samples.device)
        new_values = color_reconst.to(f_hat.dtype)
        f_hat_updated = f_hat.clone()
        f_hat_updated[samples[:, 1].int(), samples[:, 0].int()] = new_values

        f_hat = f_hat_updated

        with torch.no_grad():
            object_mask[obj_mask[:, 1].int(), obj_mask[:, 0].int(), :] = 1

            if to_vis:
                reconst_image = to_pil_image(object_mask * f_hat)
                reconst_image.save(f'./reconst_sparse_image.png')
                match_image = to_pil_image(object_mask * image_norm)
                match_image.save(f'./match_image.png')

    loss_l1 = torch.nn.L1Loss()
    reconst_loss = loss_l1(object_mask * f_hat, object_mask * image_norm)

    return reconst_loss, f_hat


def reg_laplacian_loss(mesh: TriangleMesh):
    a = igl.adjacency_list(mesh.f.cpu().numpy())

    # The actual positions
    # This call updates delta_v
    v = mesh.get_v()

    indices = []
    values = []
    # for vid, vv in enumerate(v):
    for vid in range(mesh.v_interior_count):
        neighbors = [i for i in a[vid] if i > -1 and i < mesh.v_interior_count]
        assert len(neighbors) > 0, 'No neighbors found'
        v_indices = [[vid, i] for i in neighbors]
        v_values = [1.0 / len(neighbors)] * len(neighbors)

        indices += v_indices
        values += v_values

    avg_mat = torch.sparse_coo_tensor(torch.tensor(indices).t(
    ), values, torch.Size([v.shape[0], v.shape[0]]), device=mesh.device)
    avg_delta_v = avg_mat @ mesh.delta_v

    loss_l2_sq = torch.nn.MSELoss()
    laplacian_loss = loss_l2_sq(mesh.delta_v, avg_delta_v)

    return laplacian_loss


def reg_v_laplacian_loss(mesh: TriangleMesh):
    a = igl.adjacency_list(mesh.f.cpu().numpy())

    # The actual positions
    # This call updates delta_v
    v = mesh.get_v()

    indices = []
    values = []
    # for vid, vv in enumerate(v):
    for vid in range(mesh.v_interior_count):
        neighbors = [i for i in a[vid] if i > -1 and i < mesh.v_interior_count]
        assert len(neighbors) > 0, 'No neighbors found'
        v_indices = [[vid, i] for i in neighbors]
        v_values = [1.0 / len(neighbors)] * len(neighbors)

        indices += v_indices
        values += v_values

    avg_mat = torch.sparse_coo_tensor(torch.tensor(indices).t(
    ), values, torch.Size([v.shape[0], v.shape[0]]), device=mesh.device)
    avg_delta_v = avg_mat @ mesh.get_v()

    loss_l2_sq = torch.nn.MSELoss()
    laplacian_loss = loss_l2_sq(mesh.get_v(), avg_delta_v)

    return laplacian_loss


def reg_area_loss(mesh: TriangleMesh):
    # The actual positions
    v = mesh.get_v()

    vf = v[mesh.f]
    e1 = vf[:, 1, :] - vf[:, 0, :]
    e2 = vf[:, 2, :] - vf[:, 0, :]
    face_normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(face_normals, p=2, dim=1)

    loss_l2_sq = torch.nn.MSELoss()
    area_loss = loss_l2_sq(areas, areas.mean().expand_as(areas))

    return area_loss, areas


def reg_area_change_loss(mesh: TriangleMesh):
    def compute_area(vf):
        e1 = vf[:, 1, :] - vf[:, 0, :]
        e2 = vf[:, 2, :] - vf[:, 0, :]
        face_normals = torch.cross(e1, e2, dim=1)
        areas = 0.5 * torch.norm(face_normals, p=2, dim=1)

        return areas

    init_areas = compute_area(mesh.v[mesh.f])

    # The actual positions
    v = mesh.get_v()

    vf = v[mesh.f]
    areas = compute_area(vf)

    loss_l2_sq = torch.nn.MSELoss()
    area_loss = loss_l2_sq(areas, init_areas)

    return area_loss, areas


def reg_signed_area_change_loss(mesh: TriangleMesh):
    def compute_signed_area(vf):
        e1 = vf[:, 1, :] - vf[:, 0, :]
        e2 = vf[:, 2, :] - vf[:, 0, :]
        face_normals = torch.cross(e1, e2, dim=1)
        areas = 0.5 * face_normals[:, 2]

        return areas

    init_areas = compute_signed_area(mesh.v[mesh.f])

    # The actual positions
    v = mesh.get_v()

    vf = v[mesh.f]
    areas = compute_signed_area(vf)

    loss_l2_sq = torch.nn.MSELoss()
    area_loss = loss_l2_sq(areas, init_areas)

    return area_loss, areas


def fix_boundary_box_positions(mesh: TriangleMesh):
    def find_param_by_name(model, name):
        for param_name, param in model.named_parameters():
            if param_name == name:
                return param
        return None
    delta_v = find_param_by_name(mesh, 'delta_v')
    # delta_v.grad[mesh.v_interior_count:mesh.v_interior_count+4] = 0
    delta_v.grad[mesh.v_interior_count::] = 0


def fix_boundary_vertex_positions(mesh: TriangleMesh):
    def find_param_by_name(model, name):
        for param_name, param in model.named_parameters():
            if param_name == name:
                return param
        return None
    delta_v = find_param_by_name(mesh, 'delta_v')
    delta_v.grad[mesh.boundary_vid] = 0


def set_boundary_box_positions(mesh: TriangleMesh):
    mesh.delta_v[mesh.v_interior_count::] = 0
    u = to_differential(mesh.M, mesh.v + mesh.delta_v)
    mesh.u.data = u
    mesh.delta_v = None
    mesh.get_v()


def set_boundary_vertex_positions(mesh: TriangleMesh):
    mesh.delta_v[mesh.boundary_vid] = 0
    u = to_differential(mesh.M, mesh.v + mesh.delta_v)
    mesh.u.data = u
    mesh.delta_v = None
    mesh.get_v()


def soft_boundary_vertex_loss(mesh: TriangleMesh):
    v = mesh.get_v()
    delta_v = v - mesh.v
    loss_l2_sq = torch.nn.MSELoss()
    boundary_loss = loss_l2_sq(
        delta_v[mesh.boundary_vid], torch.zeros_like(delta_v[mesh.boundary_vid]))

    return boundary_loss
