import torch
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import igl

from tools.plot_utils import plot_color_variance
from geometry.softras import color_variance_px
from geometry.softras_batch import color_variance_sampling
from geometry.meshing import triangulate
from geometry.continuous_remesh import (calc_edge_length, calc_edges,
                                        calc_face_collapses, calc_face_normals,
                                        calc_small_face_collapses, calc_sliver_collapses,
                                        calc_vertex_normals,
                                        calc_face_splits,
                                        collapse_edges, flip_edges,
                                        calc_flip_edges, calc_flip_edges_combined,
                                        pack, prepend_dummies, remove_dummies, split_edges,
                                        count_non_delaunay_edge)
from tools.geometry_utils import incircle


@torch.no_grad()
def delaunay_triangulate(V_, F_):
    vdtype = V_.dtype
    fdtype = F_.dtype
    device = V_.device

    boundary_edges = []
    loops = igl.all_boundary_loop(F_.detach().cpu().numpy())

    for l in loops:
        boundary_edges.append([])
        for i, vid in enumerate(l):
            if i > 0:
                boundary_edges[-1].append((l[i-1], l[i]))
        boundary_edges[-1].append([boundary_edges[-1]
                                  [-1][-1], boundary_edges[-1][0][0]])
    V, F = triangulate(V_, np.array(boundary_edges[0]))
    if V.shape[1] < 3:
        V = np.hstack([V, np.zeros((V.shape[0], 3 - V.shape[1]))])
    V = torch.from_numpy(V).to(vdtype).to(device)
    F = torch.from_numpy(F).to(fdtype).to(device)

    return V, F


@torch.no_grad()
def remesh_full(
        vertices_etc: torch.Tensor,  # V,D
        faces: torch.Tensor,  # F,3 long
        image: Image,
        collapse: bool,
        flip: bool,
        split_loss=2,
        max_itr=5,
        min_area=1,
        min_edge_length=1,
        harmonic_lambda=0.1,
):
    # dummies
    input_vertices_num = vertices_etc.shape[0]

    vertices_etc, faces = prepend_dummies(vertices_etc, faces)

    # collapse
    if collapse:
        for _ in range(max_itr * 2):
            vertices = vertices_etc[:, :3]  # V,3

            before_collapse_vertices_num = vertices_etc.shape[0]

            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            edge_length = calc_edge_length(vertices, edges)  # E
            face_collapse = calc_small_face_collapses(vertices, faces, edges, face_to_edge,
                                                      edge_to_face, edge_length,
                                                      min_area)
            # e[0,1] 0...ok, 1...edgelen=0
            shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
            priority = torch.where(
                face_collapse, face_collapse.float() + shortness, face_collapse.float())

            # print(f'\tCollapse: {torch.nonzero(priority).shape[0]}')
            vertices_etc, faces = collapse_edges(
                vertices_etc, faces, edges, priority)
            vertices_etc, faces = pack(vertices_etc, faces)

            after_collapse_vertices_num = vertices_etc.shape[0]
            collapsed = before_collapse_vertices_num != after_collapse_vertices_num

            if not collapsed:
                break
        for _ in range(max_itr * 2):
            vertices = vertices_etc[:, :3]  # V,3

            before_collapse_vertices_num = vertices_etc.shape[0]

            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            edge_length = calc_edge_length(vertices, edges)  # E
            face_collapse = calc_sliver_collapses(vertices, faces, edges, face_to_edge,
                                                  edge_to_face, edge_length,
                                                  max_angle=120/180*torch.pi)
            # e[0,1] 0...ok, 1...edgelen=0
            shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
            priority = torch.where(
                face_collapse, face_collapse.float() + shortness, face_collapse.float())

            # print(f'\tSliver collapse: {torch.nonzero(priority).shape[0]}')
            vertices_etc, faces = collapse_edges(
                vertices_etc, faces, edges, priority)
            vertices_etc, faces = pack(vertices_etc, faces)

            after_collapse_vertices_num = vertices_etc.shape[0]
            collapsed = before_collapse_vertices_num != after_collapse_vertices_num

            if not collapsed:
                break

    if split_loss > 0:
        for _ in range(int(max_itr / 2)):
            vertices_etc, faces, split_edge = calc_face_splits(
                vertices_etc, faces, edges, face_to_edge, image, edge_length, split_loss, min_area)
            vertices_etc, faces = pack(vertices_etc, faces)

            if split_edge == 0:
                break

    vertices = vertices_etc[:, :3]

    # Delaunary flip
    V, F = remove_dummies(vertices_etc, faces)
    V, F = delaunay_triangulate(V, F)
    vertices_etc, faces = prepend_dummies(V, F)

    if flip:
        track_non_delaunay = False
        for itr in range(5):
            vertices = vertices_etc[:, :3]
            num_flipped = 0
            flip_edge_to_face = None

            _, f_color_var = color_variance_sampling(
                vertices[1:], faces[1:]-1, image)

            f_color_var = torch.concat(
                (torch.zeros((1), dtype=torch.long, device=f_color_var.device), f_color_var), dim=0)
            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            flip_vis_itr = -1
            num_flipped, flip_edge_to_face = calc_flip_edges_combined(vertices, faces, face_to_edge,
                                                                      edges, edge_to_face, f_color_var, image,
                                                                      with_normal_check=True,
                                                                      harmonic_lambda=harmonic_lambda,
                                                                      vis_itr=flip_vis_itr)
            # print(f'\tFlipped: {num_flipped}')

            if track_non_delaunay:
                edges, face_to_edge, edge_to_face = calc_edges(
                    faces, with_edge_to_face=True)  # E,2 F,3
                E = edges.shape[0]
                non_delaunay = count_non_delaunay_edge(faces, vertices)

                print(f'\tNon-Delaunay: {non_delaunay}/{E}')

            if num_flipped == 0:
                break

    # Remove near degenerate triangles
    if collapse:
        # if False:
        # small_min_area = 1e-3
        small_min_area = 5e-4
        for _ in range(max_itr):
            vertices = vertices_etc[:, :3]  # V,3

            before_collapse_vertices_num = vertices_etc.shape[0]

            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            edge_length = calc_edge_length(vertices, edges)  # E
            face_collapse = calc_small_face_collapses(vertices, faces, edges, face_to_edge,
                                                      edge_to_face, edge_length,
                                                      small_min_area)
            # e[0,1] 0...ok, 1...edgelen=0
            shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
            priority = torch.where(
                face_collapse, face_collapse.float() + shortness, face_collapse.float())

            # print(f'\tFinal Collapse: {torch.nonzero(priority).shape[0]}')
            vertices_etc, faces = collapse_edges(
                vertices_etc, faces, edges, priority)
            vertices_etc, faces = pack(vertices_etc, faces)

            after_collapse_vertices_num = vertices_etc.shape[0]
            collapsed = before_collapse_vertices_num != after_collapse_vertices_num

            if not collapsed:
                break

    V, F = remove_dummies(vertices_etc, faces)

    # print(f'Remesh: {input_vertices_num} -> {V.shape[0]}')

    return V, F


@torch.no_grad()
def remesh_final_clean(
        vertices_etc: torch.Tensor,  # V,D
        faces: torch.Tensor,  # F,3 long
        max_itr=5,
        min_area=1,
        min_edge_length=1,
):
    # dummies
    input_vertices_num = vertices_etc.shape[0]

    vertices_etc, faces = prepend_dummies(vertices_etc, faces)

    # Remove near degenerate triangles
    small_min_area = min_area
    for _ in range(max_itr):
        vertices = vertices_etc[:, :3]  # V,3

        before_collapse_vertices_num = vertices_etc.shape[0]

        edges, face_to_edge, edge_to_face = calc_edges(
            faces, with_edge_to_face=True)  # E,2 F,3
        edge_length = calc_edge_length(vertices, edges)  # E
        face_collapse = calc_small_face_collapses(vertices, faces, edges, face_to_edge,
                                                  edge_to_face, edge_length,
                                                  small_min_area)
        # e[0,1] 0...ok, 1...edgelen=0
        shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
        priority = torch.where(
            face_collapse, face_collapse.float() + shortness, face_collapse.float())

        # print(f'\tFinal Collapse: {torch.nonzero(priority).shape[0]}')
        vertices_etc, faces = collapse_edges(
            vertices_etc, faces, edges, priority)
        vertices_etc, faces = pack(vertices_etc, faces)

        after_collapse_vertices_num = vertices_etc.shape[0]
        collapsed = before_collapse_vertices_num != after_collapse_vertices_num

        if not collapsed:
            break

    for _ in range(max_itr * 2):
        vertices = vertices_etc[:, :3]  # V,3

        before_collapse_vertices_num = vertices_etc.shape[0]

        edges, face_to_edge, edge_to_face = calc_edges(
            faces, with_edge_to_face=True)  # E,2 F,3
        edge_length = calc_edge_length(vertices, edges)  # E
        face_collapse = calc_sliver_collapses(vertices, faces, edges, face_to_edge,
                                              edge_to_face, edge_length,
                                              max_angle=120/180*torch.pi)
        # e[0,1] 0...ok, 1...edgelen=0
        shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
        priority = torch.where(
            face_collapse, face_collapse.float() + shortness, face_collapse.float())

        # print(
        #     f'\tFinal sliver collapse: {torch.nonzero(priority).shape[0]}')
        vertices_etc, faces = collapse_edges(
            vertices_etc, faces, edges, priority)
        vertices_etc, faces = pack(vertices_etc, faces)

        after_collapse_vertices_num = vertices_etc.shape[0]
        collapsed = before_collapse_vertices_num != after_collapse_vertices_num

        if not collapsed:
            break

    V, F = remove_dummies(vertices_etc, faces)

    # print(f'Remesh: {input_vertices_num} -> {V.shape[0]}')

    return V, F


@torch.no_grad()
def remesh_full_combined(
        vertices_etc: torch.Tensor,  # V,D
        faces: torch.Tensor,  # F,3 long
        image: Image,
        collapse: bool,
        flip: bool,
        split_loss=2,
        max_itr=5,
        min_area=1,
        min_edge_length=1,
        harmonic_lambda=0.1,
        vis_itr=0,
):
    # dummies
    input_vertices_num = vertices_etc.shape[0]

    vertices_etc, faces = prepend_dummies(vertices_etc, faces)

    # collapse
    if collapse:
        for _ in range(max_itr * 2):
            vertices = vertices_etc[:, :3]  # V,3

            before_collapse_vertices_num = vertices_etc.shape[0]

            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            edge_length = calc_edge_length(vertices, edges)  # E
            face_collapse = calc_small_face_collapses(vertices, faces, edges, face_to_edge,
                                                      edge_to_face, edge_length,
                                                      min_area)
            # e[0,1] 0...ok, 1...edgelen=0
            shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
            priority = torch.where(
                face_collapse, face_collapse.float() + shortness, face_collapse.float())

            # print(f'\tCollapse: {torch.nonzero(priority).shape[0]}')
            vertices_etc, faces = collapse_edges(
                vertices_etc, faces, edges, priority)
            vertices_etc, faces = pack(vertices_etc, faces)

            after_collapse_vertices_num = vertices_etc.shape[0]
            collapsed = before_collapse_vertices_num != after_collapse_vertices_num

            if not collapsed:
                break

        for _ in range(max_itr * 2):
            vertices = vertices_etc[:, :3]  # V,3

            before_collapse_vertices_num = vertices_etc.shape[0]

            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            edge_length = calc_edge_length(vertices, edges)  # E
            face_collapse = calc_sliver_collapses(vertices, faces, edges, face_to_edge,
                                                  edge_to_face, edge_length,
                                                  max_angle=120/180*torch.pi)
            # e[0,1] 0...ok, 1...edgelen=0
            shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
            priority = torch.where(
                face_collapse, face_collapse.float() + shortness, face_collapse.float())

            # print(f'\tSliver collapse: {torch.nonzero(priority).shape[0]}')
            vertices_etc, faces = collapse_edges(
                vertices_etc, faces, edges, priority)
            vertices_etc, faces = pack(vertices_etc, faces)

            after_collapse_vertices_num = vertices_etc.shape[0]
            collapsed = before_collapse_vertices_num != after_collapse_vertices_num

            if not collapsed:
                break

    if split_loss > 0:
        for _ in range(int(max_itr / 2)):
            # # _, f_color_var = color_variance_px(
            # #     vertices[1:], faces[1:]-1, image, face_mask=None)
            # _, f_color_var = color_variance_sampling(
            #     vertices[1:], faces[1:]-1, image)
            # edges, face_to_edge = calc_edges(faces)  # E,2 F,3
            # edge_length = calc_edge_length(vertices, edges)  # E
            # face_to_split = torch.cat(
            #     (torch.zeros(1, dtype=bool, device=f_color_var.device), f_color_var > split_loss), 0)
            # local_edge_id_to_split = torch.argmax(
            #     edge_length[face_to_edge[face_to_split]], 1)
            # edge_to_split = face_to_edge[face_to_split, local_edge_id_to_split]
            # splits = torch.zeros(
            #     edges.shape[0], dtype=torch.bool, device=vertices.device)
            # splits[edge_to_split] = True
            # print(f'\tSplit: {edge_to_split.shape[0]}')

            # vertices_etc, faces = split_edges(
            #     vertices_etc, faces, edges, face_to_edge, splits, pack_faces=False)
            vertices_etc, faces, split_edge = calc_face_splits(
                vertices_etc, faces, edges, face_to_edge, image, edge_length, split_loss, min_area)
            vertices_etc, faces = pack(vertices_etc, faces)

            if split_edge == 0:
                break

    if flip:
        track_non_delaunay = False
        for itr in range(max_itr):
            vertices = vertices_etc[:, :3]
            num_flipped = 0
            flip_edge_to_face = None

            # _, f_color_var = color_variance_px(
            #     vertices[1:], faces[1:]-1, image, face_mask=None)
            _, f_color_var = color_variance_sampling(
                vertices[1:], faces[1:]-1, image)

            # ax = plot_color_variance(
            #     vertices[1:], faces[1:]-1, image, f_color_var.detach().cpu().numpy())
            # png_name = f'./remeshing_color.svg'
            # plt.savefig(png_name, dpi=600)
            # plt.close()

            f_color_var = torch.concat(
                (torch.zeros((1), dtype=torch.long, device=f_color_var.device), f_color_var), dim=0)
            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            flip_vis_itr = -1 if vis_itr < 0 else (vis_itr*100 + itr)
            num_flipped, flip_edge_to_face = calc_flip_edges_combined(vertices, faces, face_to_edge,
                                                                      edges, edge_to_face, f_color_var, image,
                                                                      with_normal_check=True,
                                                                      harmonic_lambda=harmonic_lambda,
                                                                      vis_itr=flip_vis_itr)
            # print(f'\tFlipped: {num_flipped}')

            if track_non_delaunay:
                edges, face_to_edge, edge_to_face = calc_edges(
                    faces, with_edge_to_face=True)  # E,2 F,3
                E = edges.shape[0]
                non_delaunay = count_non_delaunay_edge(faces, vertices)

                print(f'\tNon-Delaunay: {non_delaunay}/{E}')

            if num_flipped == 0:
                break

    # Remove near degenerate triangles
    if collapse:
        small_min_area = 1e-3
        for _ in range(max_itr):
            vertices = vertices_etc[:, :3]  # V,3

            before_collapse_vertices_num = vertices_etc.shape[0]

            edges, face_to_edge, edge_to_face = calc_edges(
                faces, with_edge_to_face=True)  # E,2 F,3
            edge_length = calc_edge_length(vertices, edges)  # E
            face_collapse = calc_small_face_collapses(vertices, faces, edges, face_to_edge,
                                                      edge_to_face, edge_length,
                                                      small_min_area)
            # e[0,1] 0...ok, 1...edgelen=0
            shortness = (1 - edge_length / min_edge_length).clamp_min_(0)
            priority = torch.where(
                face_collapse, face_collapse.float() + shortness, face_collapse.float())

            # print(f'\tFinal Collapse: {torch.nonzero(priority).shape[0]}')
            vertices_etc, faces = collapse_edges(
                vertices_etc, faces, edges, priority)
            vertices_etc, faces = pack(vertices_etc, faces)

            after_collapse_vertices_num = vertices_etc.shape[0]
            collapsed = before_collapse_vertices_num != after_collapse_vertices_num

            if not collapsed:
                break

    V, F = remove_dummies(vertices_etc, faces)

    # print(f'Remesh: {input_vertices_num} -> {V.shape[0]}')

    return V, F
