from pathlib import Path

import igl
import math
import torch
import torch.nn.functional as tfunc
import torch_scatter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from geometry.softras_batch import color_variance_sampling
from tools.plot_utils import plot_color_variance, plot_VF
from tools.geometry_utils import incircle, compute_angles_atan2


def count_non_delaunay_edge(dummy_faces, dummy_vertices):
    edges, _, edge_to_face = calc_edges(
        dummy_faces, with_edge_to_face=True)  # E,2 F,3
    E = edges.shape[0]
    orig_faces = dummy_faces[1:] - 1
    vertices = dummy_vertices[1::]
    vertices_norm = vertices
    non_delaunay = 0
    for ei in range(E):
        f_neighbors = edge_to_face[ei, :, 0] - 1
        f_vv = orig_faces[f_neighbors].clone()
        vertices_not_shared = []
        for vi in range(f_vv.size(0)):
            unique_vertex = [ii for ii, vertex in enumerate(f_vv[vi])
                             if vertex not in f_vv[1-vi]]
            if len(unique_vertex) == 0:
                continue
            vertices_not_shared.append(unique_vertex[0])
        if len(vertices_not_shared) < 2:
            continue

        # Test if any of the two adjacent triangles is in circle
        incircle_0 = incircle(vertices_norm[f_vv[1, 0]].detach().cpu().numpy(),
                              vertices_norm[f_vv[1, 1]
                                            ].detach().cpu().numpy(),
                              vertices_norm[f_vv[1, 2]
                                            ].detach().cpu().numpy(),
                              vertices_norm[f_vv[0, vertices_not_shared[0]]].detach().cpu().numpy())
        incircle_1 = incircle(vertices_norm[f_vv[0, 0]].detach().cpu().numpy(),
                              vertices_norm[f_vv[0, 1]
                                            ].detach().cpu().numpy(),
                              vertices_norm[f_vv[0, 2]
                                            ].detach().cpu().numpy(),
                              vertices_norm[f_vv[1, vertices_not_shared[1]]].detach().cpu().numpy())
        if (incircle_0 > 0) or (incircle_1 > 0):
            non_delaunay += 1

    return non_delaunay


def prepend_dummies(
    vertices: torch.Tensor,  # V,D
    faces: torch.Tensor,  # F,3 long
):
    """prepend dummy elements to vertices and faces to enable "masked" scatter operations"""
    V, D = vertices.shape
    vertices = torch.concat((torch.full(
        (1, D), fill_value=torch.nan, device=vertices.device), vertices), dim=0)
    faces = torch.concat(
        (torch.zeros((1, 3), dtype=torch.long, device=faces.device), faces+1), dim=0)
    return vertices, faces


def remove_dummies(
    vertices: torch.Tensor,  # V,D - first vertex all nan and unreferenced
    faces: torch.Tensor,  # F,3 long - first face all zeros
):
    """remove dummy elements added with prepend_dummies()"""
    return vertices[1:], faces[1:]-1


def calc_edges(
    faces: torch.Tensor,  # F,3 long - first face may be dummy with all zeros
    with_edge_to_face: bool = False
):
    """
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |      
    e0---->-o     
    """

    F = faces.shape[0]

    # make full edges, lower vertex index first
    face_edges = torch.stack((faces, faces.roll(-1, 1)), dim=-1)  # F*3,3,2
    full_edges = face_edges.reshape(F*3, 2)
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2 TODO min/max faster?

    # make unique edges
    edges, full_to_unique = torch.unique(
        input=sorted_edges, sorted=True, return_inverse=True, dim=0)  # (E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F, 3)  # F,3

    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:, 0] != sorted_edges[:, 0]  # F*3
    edge_to_face = torch.zeros(
        (E, 2, 2), dtype=torch.long, device=faces.device)  # E,LR=2,S=2
    scatter_src = torch.cartesian_prod(torch.arange(
        0, F, device=faces.device), torch.arange(0, 3, device=faces.device))  # F*3,2
    edge_to_face.reshape(2*E, 2).scatter_(dim=0, index=(2*full_to_unique+is_right)
                                          [:, None].expand(F*3, 2), src=scatter_src)  # E,LR=2,S=2
    edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face


def calc_edge_length(
        vertices: torch.Tensor,  # V,3 first may be dummy
        # E,2 long, lower vertex index first, (0,0) for unused
        edges: torch.Tensor,
):  # E

    full_vertices = vertices[edges]  # E,2,3
    a, b = full_vertices.unbind(dim=1)  # E,3
    return torch.norm(a-b, p=2, dim=-1)


def calc_face_normals(
        vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
        faces: torch.Tensor,  # F,3 long, first face may be all zero
        normalize: bool = False,
):  # F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces]  # F,C=3,3
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1-v0, v2-v0, dim=1)  # F,3
    if normalize:
        face_normals = tfunc.normalize(
            face_normals, eps=1e-6, dim=1)  # TODO inplace?
    return face_normals  # F,3


def calc_vertex_normals(
        vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
        faces: torch.Tensor,  # F,3 long, first face may be all zero
        face_normals: torch.Tensor = None,  # F,3, not normalized
):  # F,3

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)

    vertex_normals = torch.zeros(
        (vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device)  # V,C=3,3
    vertex_normals.scatter_add_(dim=0, index=faces[:, :, None].expand(
        F, 3, 3), src=face_normals[:, None, :].expand(F, 3, 3))
    vertex_normals = vertex_normals.sum(dim=1)  # V,3
    return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)


def calc_face_ref_normals(
        faces: torch.Tensor,  # F,3 long, 0 for unused
        vertex_normals: torch.Tensor,  # V,3 first unused
        normalize: bool = False,
):  # F,3
    """calculate reference normals for face flip detection"""
    full_normals = vertex_normals[faces]  # F,C=3,3
    ref_normals = full_normals.sum(dim=1)  # F,3
    if normalize:
        ref_normals = tfunc.normalize(ref_normals, eps=1e-6, dim=1)
    return ref_normals


def pack(
        vertices: torch.Tensor,  # V,3 first unused and nan
        faces: torch.Tensor,  # F,3 long, 0 for unused
):  # (vertices,faces), keeps first vertex unused
    """removes unused elements in vertices and faces"""
    V = vertices.shape[0]

    # remove unused faces
    used_faces = faces[:, 0] != 0
    used_faces[0] = True
    faces = faces[used_faces]  # sync

    # remove unused vertices
    used_vertices = torch.zeros(V, 3, dtype=torch.bool, device=vertices.device)
    used_vertices.scatter_(dim=0, index=faces, value=True,
                           reduce='add')  # TODO int faster?
    used_vertices = used_vertices.any(dim=1)
    used_vertices[0] = True
    vertices = vertices[used_vertices]  # sync

    # update used faces
    ind = torch.zeros(V, dtype=torch.long, device=vertices.device)
    V1 = used_vertices.sum()
    ind[used_vertices] = torch.arange(0, V1, device=vertices.device)  # sync
    faces = ind[faces]

    return vertices, faces


def split_edges(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        splits,  # E bool
        pack_faces: bool = True,
):  # (vertices,faces)

    #   c2                    c2               c...corners = faces
    #    . .                   . .             s...side_vert, 0 means no split
    #    .   .                 .N2 .           S...shrunk_face
    #    .     .               .     .         Ni...new_faces
    #   s2      s1           s2|c2...s1|c1
    #    .        .            .     .  .
    #    .          .          . S .      .
    #    .            .        . .     N1    .
    #   c0...(s0=0)....c1    s0|c0...........c1
    #
    # pseudo-code:
    #   S = [s0|c0,s1|c1,s2|c2] example:[c0,s1,s2]
    #   split = side_vert!=0 example:[False,True,True]
    #   N0 = split[0]*[c0,s0,s2|c2] example:[0,0,0]
    #   N1 = split[1]*[c1,s1,s0|c0] example:[c1,s1,c0]
    #   N2 = split[2]*[c2,s2,s1|c1] example:[c2,s2,s1]

    V = vertices.shape[0]
    F = faces.shape[0]
    S = splits.sum().item()  # sync

    if S == 0:
        return vertices, faces

    edge_vert = torch.zeros_like(splits, dtype=torch.long)  # E
    edge_vert[splits] = torch.arange(
        V, V+S, dtype=torch.long, device=vertices.device)  # E 0 for no split, sync
    side_vert = edge_vert[face_to_edge]  # F,3 long, 0 for no split
    split_edges = edges[splits]  # S sync

    # vertices
    split_vertices = vertices[split_edges].mean(dim=1)  # S,3
    vertices = torch.concat((vertices, split_vertices), dim=0)

    # faces
    side_split = side_vert != 0  # F,3
    # F,3 long, 0 for no split
    shrunk_faces = torch.where(side_split, side_vert, faces)
    new_faces = side_split[:, :, None] * torch.stack(
        (faces, side_vert, shrunk_faces.roll(1, dims=-1)), dim=-1)  # F,N=3,C=3
    faces = torch.concat((shrunk_faces, new_faces.reshape(F*3, 3)))  # 4F,3
    if pack_faces:
        mask = faces[:, 0] != 0
        mask[0] = True
        faces = faces[mask]  # F',3 sync

    return vertices, faces


def collapse_edges(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        priorities: torch.Tensor,  # E float
        stable: bool = False,  # only for unit testing
):  # (vertices,faces)

    V = vertices.shape[0]

    # check spacing
    _, order = priorities.sort(stable=stable)  # E
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0, len(rank), device=rank.device)
    vert_rank = torch.zeros(V, dtype=torch.long, device=vertices.device)  # V
    edge_rank = rank  # E
    for i in range(3):
        torch_scatter.scatter_max(src=edge_rank[:, None].expand(
            -1, 2).reshape(-1), index=edges.reshape(-1), dim=0, out=vert_rank)
        edge_rank, _ = vert_rank[edges].max(dim=-1)  # E
    candidates = edges[(edge_rank == rank).logical_and_(
        priorities > 0)]  # E',2

    # check connectivity
    vert_connections = torch.zeros(
        V, dtype=torch.long, device=vertices.device)  # V
    vert_connections[candidates[:, 0]] = 1  # start
    edge_connections = vert_connections[edges].sum(
        dim=-1)  # E, edge connected to start
    vert_connections.scatter_add_(dim=0, index=edges.reshape(
        -1), src=edge_connections[:, None].expand(-1, 2).reshape(-1))  # one edge from start
    vert_connections[candidates] = 0  # clear start and end
    edge_connections = vert_connections[edges].sum(
        dim=-1)  # E, one or two edges from start
    vert_connections.scatter_add_(dim=0, index=edges.reshape(
        -1), src=edge_connections[:, None].expand(-1, 2).reshape(-1))  # one or two edges from start
    # E" not more than two connections between start and end
    collapses = candidates[vert_connections[candidates[:, 1]] <= 2]

    # mean vertices
    vertices[collapses[:, 0]] = vertices[collapses].mean(dim=1)  # TODO dim?

    # update faces
    dest = torch.arange(0, V, dtype=torch.long, device=vertices.device)  # V
    dest[collapses[:, 1]] = dest[collapses[:, 0]]
    faces = dest[faces]  # F,3 TODO optimize?
    c0, c1, c2 = faces.unbind(dim=-1)
    collapsed = (c0 == c1).logical_or_(c1 == c2).logical_or_(c0 == c2)
    faces[collapsed] = 0

    return vertices, faces


def calc_face_collapses(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        edge_length: torch.Tensor,  # E
        face_normals: torch.Tensor,  # F,3
        vertex_normals: torch.Tensor,  # V,3 first unused
        min_edge_length: torch.Tensor = None,  # V
        area_ratio=0.5,  # collapse if area < min_edge_length**2 * area_ratio
        shortest_probability=0.8
):  # E edges to collapse

    E = edges.shape[0]
    F = faces.shape[0]

    # face flips
    ref_normals = calc_face_ref_normals(
        faces, vertex_normals, normalize=False)  # F,3
    face_collapses = (face_normals*ref_normals).sum(dim=-1) < 0  # F

    # small faces
    if min_edge_length is not None:
        min_face_length = min_edge_length[faces].mean(dim=-1)  # F
        min_area = min_face_length**2 * area_ratio  # F
        face_collapses.logical_or_(face_normals.norm(dim=-1) < min_area*2)  # F
        face_collapses[0] = False

    # faces to edges
    face_length = edge_length[face_to_edge]  # F,3

    if shortest_probability < 1:
        # select shortest edge with shortest_probability chance
        randlim = round(2/(1-shortest_probability))
        rand_ind = torch.randint(0, randlim, size=(F,), device=faces.device).clamp_max_(
            2)  # selected edge local index in face
        sort_ind = torch.argsort(face_length, dim=-1, descending=True)  # F,3
        local_ind = sort_ind.gather(dim=-1, index=rand_ind[:, None])
    else:
        # F,1 0...2 shortest edge local index in face
        local_ind = torch.argmin(face_length, dim=-1)[:, None]

    edge_ind = face_to_edge.gather(dim=1, index=local_ind)[
        :, 0]  # F 0...E selected edge global index
    edge_collapses = torch.zeros(E, dtype=torch.long, device=vertices.device)
    edge_collapses.scatter_add_(
        dim=0, index=edge_ind, src=face_collapses.long())  # TODO legal for bool?

    return edge_collapses.bool()


def calc_small_face_collapses(
        v: torch.Tensor,  # V,3 first unused
        f: torch.Tensor,  # F,3 long, 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        edge_to_face: torch.Tensor,  # E,[left,right],[face,side]
        edge_length: torch.Tensor,  # E
        min_area=1
):  # E edges to collapse
    E = edges.shape[0]

    vf = v[f]
    e1 = vf[:, 1, :] - vf[:, 0, :]
    e2 = vf[:, 2, :] - vf[:, 0, :]
    face_normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(face_normals, p=2, dim=1)

    face_collapses = areas < min_area

    # Remove the faces that are on boundary
    neighbor_corner = (edge_to_face[:, :, 1] + 2) % 3  # go from side to corner
    neighbors = f[edge_to_face[:, :, 0], neighbor_corner]  # E,LR=2
    edge_is_boundary = ~neighbors.all(dim=-1)  # E
    boundary_vertices = edges[edge_is_boundary].ravel().unique()
    boundary_faces = (f[..., None] == boundary_vertices).any(dim=1).any(dim=1)
    # boundary_faces = edge_to_face[edge_is_boundary, :, 0].ravel().unique()
    face_collapses[boundary_faces] = False


    # faces to edges
    face_length = edge_length[face_to_edge]  # F,3
    # F,1 0...2 shortest edge local index in face
    local_ind = torch.argmin(face_length, dim=-1)[:, None]

    edge_ind = face_to_edge.gather(dim=1, index=local_ind)[
        :, 0]  # F 0...E selected edge global index
    edge_collapses = torch.zeros(E, dtype=torch.long, device=v.device)
    edge_collapses.scatter_add_(
        dim=0, index=edge_ind, src=face_collapses.long())  # TODO legal for bool?

    return edge_collapses.bool()


def calc_sliver_collapses(
        v: torch.Tensor,  # V,3 first unused
        f: torch.Tensor,  # F,3 long, 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        edge_to_face: torch.Tensor,  # E,[left,right],[face,side]
        edge_length: torch.Tensor,  # E
        max_angle=120/180*torch.pi
):  # E edges to collapse
    E = edges.shape[0]

    vf = v[f]

    def triangle_angles(idx):
        e1 = vf[:, (idx+1) % 3, :] - vf[:, idx, :]
        e2 = vf[:, (idx+2) % 3, :] - vf[:, idx, :]
        angles = compute_angles_atan2(e2, e1, set_ref=False)
        # angles = 2*torch.pi - angles

        return angles

    face_max_angles = torch.stack([triangle_angles(i)
                                  for i in range(3)], dim=1)
    face_max_angles = face_max_angles.max(dim=1).values

    face_collapses = face_max_angles > max_angle

    # Remove the faces that are on boundary
    neighbor_corner = (edge_to_face[:, :, 1] + 2) % 3  # go from side to corner
    neighbors = f[edge_to_face[:, :, 0], neighbor_corner]  # E,LR=2
    edge_is_boundary = ~neighbors.all(dim=-1)  # E
    boundary_vertices = edges[edge_is_boundary].ravel().unique()
    boundary_faces = (f[..., None] == boundary_vertices).any(dim=1).any(dim=1)
    # boundary_faces = edge_to_face[edge_is_boundary, :, 0].ravel().unique()
    face_collapses[boundary_faces] = False


    # faces to edges
    face_length = edge_length[face_to_edge]  # F,3
    # F,1 0...2 shortest edge local index in face
    local_ind = torch.argmin(face_length, dim=-1)[:, None]

    edge_ind = face_to_edge.gather(dim=1, index=local_ind)[
        :, 0]  # F 0...E selected edge global index
    edge_collapses = torch.zeros(E, dtype=torch.long, device=v.device)
    edge_collapses.scatter_add_(
        dim=0, index=edge_ind, src=face_collapses.long())  # TODO legal for bool?

    return edge_collapses.bool()


def calc_face_splits(
        vertices_etc: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, 0 for unused
        edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        image: Image,
        edge_length: torch.Tensor,  # E
        split_loss: float,
        min_area: float
):
    vertices = vertices_etc[:, :3]  # V,3

    vf = vertices[faces]
    e1 = vf[:, 1, :] - vf[:, 0, :]
    e2 = vf[:, 2, :] - vf[:, 0, :]
    face_normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(face_normals, p=2, dim=1)

    face_candidates = areas >= min_area

    _, f_color_var = color_variance_sampling(vertices[1:], faces[1:]-1, image)
    edges, face_to_edge = calc_edges(faces)  # E,2 F,3
    edge_length = calc_edge_length(vertices, edges)  # E
    face_to_split = torch.cat(
        (torch.zeros(1, dtype=bool, device=f_color_var.device), f_color_var > split_loss), 0)

    face_to_split = face_candidates.logical_and_(face_to_split)

    local_edge_id_to_split = torch.argmax(
        edge_length[face_to_edge[face_to_split]], 1)
    edge_to_split = face_to_edge[face_to_split, local_edge_id_to_split]
    splits = torch.zeros(
        edges.shape[0], dtype=torch.bool, device=vertices.device)
    splits[edge_to_split] = True
    print(f'\tSplit: {edge_to_split.shape[0]}')

    vertices_etc, faces = split_edges(
        vertices_etc, faces, edges, face_to_edge, splits, pack_faces=False)

    return vertices_etc, faces, edge_to_split.shape[0]


def flip_edges(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, first must be 0, 0 for unused
        edges: torch.Tensor,  # E,2 long, first must be 0, 0 for unused, lower vertex index first
        edge_to_face: torch.Tensor,  # E,[left,right],[face,side]
        with_border: bool = True,  # handle border edges (D=4 instead of D=6)
        with_normal_check: bool = True,  # check face normal flips
        stable: bool = False,  # only for unit testing
):
    V = vertices.shape[0]
    E = edges.shape[0]
    device = vertices.device
    vertex_degree = torch.zeros(V, dtype=torch.long, device=device)  # V long
    vertex_degree.scatter_(
        dim=0, index=edges.reshape(E*2), value=1, reduce='add')
    neighbor_corner = (edge_to_face[:, :, 1] + 2) % 3  # go from side to corner
    neighbors = faces[edge_to_face[:, :, 0], neighbor_corner]  # E,LR=2
    edge_is_inside = neighbors.all(dim=-1)  # E

    if with_border:
        # inside vertices should have D=6, border edges D=4, so we subtract 2 for all inside vertices
        # need to use float for masks in order to use scatter(reduce='multiply')
        vertex_is_inside = torch.ones(
            V, 2, dtype=torch.float32, device=vertices.device)  # V,2 float
        src = edge_is_inside.type(torch.float32)[
            :, None].expand(E, 2)  # E,2 float
        vertex_is_inside.scatter_(
            dim=0, index=edges, src=src, reduce='multiply')
        vertex_is_inside = vertex_is_inside.prod(
            dim=-1, dtype=torch.long)  # V long
        vertex_degree -= 2 * vertex_is_inside  # V long

    neighbor_degrees = vertex_degree[neighbors]  # E,LR=2
    edge_degrees = vertex_degree[edges]  # E,2
    #
    # loss = Sum_over_affected_vertices((new_degree-6)**2)
    # loss_change = Sum_over_neighbor_vertices((degree+1-6)**2-(degree-6)**2)
    #                   + Sum_over_edge_vertices((degree-1-6)**2-(degree-6)**2)
    #             = 2 * (2 + Sum_over_neighbor_vertices(degree) - Sum_over_edge_vertices(degree))
    #
    loss_change = 2 + \
        neighbor_degrees.sum(dim=-1) - edge_degrees.sum(dim=-1)  # E
    candidates = torch.logical_and(loss_change < 0, edge_is_inside)  # E
    loss_change = loss_change[candidates]  # E'
    if loss_change.shape[0] == 0:
        return

    edges_neighbors = torch.concat(
        (edges[candidates], neighbors[candidates]), dim=-1)  # E',4
    _, order = loss_change.sort(descending=True, stable=stable)  # E'
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0, len(rank), device=rank.device)
    vertex_rank = torch.zeros((V, 4), dtype=torch.long, device=device)  # V,4
    torch_scatter.scatter_max(
        src=rank[:, None].expand(-1, 4), index=edges_neighbors, dim=0, out=vertex_rank)
    vertex_rank, _ = vertex_rank.max(dim=-1)  # V
    neighborhood_rank, _ = vertex_rank[edges_neighbors].max(dim=-1)  # E'
    flip = rank == neighborhood_rank  # E'

    if with_normal_check:
        #  cl-<-----e1     e0,e1...edge, e0<e1
        #   |      /A      L,R....left and right face
        #   |  L /  |      both triangles ordered counter clockwise
        #   |  / R  |      normals pointing out of screen
        #   V/      |
        #   e0---->-cr
        v = vertices[edges_neighbors]  # E",4,3
        v = v - v[:, 0:1]  # make relative to e0
        e1 = v[:, 1]
        cl = v[:, 2]
        cr = v[:, 3]
        # sum of old normal vectors
        n = torch.cross(e1, cl) + torch.cross(cr, e1)
        flip.logical_and_(torch.sum(n*torch.cross(cr, cl),
                          dim=-1) > 0)  # first new face
        # second new face
        flip.logical_and_(torch.sum(n*torch.cross(cl-e1, cr-e1), dim=-1) > 0)

    flip_edges_neighbors = edges_neighbors[flip]  # E",4
    flip_edge_to_face = edge_to_face[candidates, :, 0][flip]  # E",2
    flip_faces = flip_edges_neighbors[:, [[0, 3, 2], [1, 2, 3]]]  # E",2,3
    faces.scatter_(dim=0, index=flip_edge_to_face.reshape(-1,
                   1).expand(-1, 3), src=flip_faces.reshape(-1, 3))


def calc_flip_edges(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, first must be 0, 0 for unused
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        edges: torch.Tensor,  # E,2 long, first must be 0, 0 for unused, lower vertex index first
        edge_to_face: torch.Tensor,  # E,[left,right],[face,side]
        f_color_var: torch.Tensor,
        image: Image,
        with_normal_check: bool = True,  # check face normal flips
        stable: bool = False,  # only for unit testing
        vis_itr: int = -1
):
    V = vertices.shape[0]
    E = edges.shape[0]
    device = vertices.device
    neighbor_corner = (edge_to_face[:, :, 1] + 2) % 3  # go from side to corner
    neighbors = faces[edge_to_face[:, :, 0], neighbor_corner]  # E,LR=2
    edge_is_inside = neighbors.all(dim=-1)  # E

    # The edge can't be in a boundary face
    # edge_is_boundary = ~neighbors.all(dim=-1)  # E
    # boundary_faces = edge_to_face[edge_is_boundary, :, 0].ravel().unique()
    # boundary_neighboring_edges = face_to_edge[boundary_faces].ravel().unique()

    # Find the current color variance of the two adjacent faces
    e_color_var = f_color_var[edge_to_face[:, :, 0]].sum(axis=1)

    # Calculate the color variance of the new faces
    orig_faces = faces[1:] - 1
    e_color_var_after = e_color_var.clone()
    loss_epsilon = 1
    flip_loss_epsilon = 1
    non_overlapping_edges = {
        'ff_flipped': [[]],
        'dst_ei': [[]],
        'neighbor_edges': [set()],
    }
    for ei in range(E):
        if not edge_is_inside[ei] or e_color_var[ei] < loss_epsilon:
            # if not edge_is_inside[ei]:
            continue
        f_neighbors = edge_to_face[ei, :, 0] - 1
        f_vv = orig_faces[f_neighbors]
        vertices_not_shared = []
        for vi in range(f_vv.size(0)):
            unique_vertex = [ii for ii, vertex in enumerate(f_vv[vi])
                             if vertex not in f_vv[1-vi]]
            if len(unique_vertex) == 0:
                continue
            vertices_not_shared.append(unique_vertex[0])
        if len(vertices_not_shared) < 2:
            continue
        f_vv_flipped = torch.tensor([[f_vv[0, vertices_not_shared[0]], f_vv[0, (vertices_not_shared[0] + 1) % 3], f_vv[1, vertices_not_shared[1]]],
                                     [f_vv[1, vertices_not_shared[1]], f_vv[1, (vertices_not_shared[1] + 1) % 3], f_vv[0, vertices_not_shared[0]]]], device=faces.device)

        ei_added = False
        n_edges = face_to_edge[edge_to_face[ei, :, 0]]
        for ii in range(len(non_overlapping_edges['neighbor_edges'])):
            if ei not in non_overlapping_edges['neighbor_edges'][ii]:
                non_overlapping_edges['ff_flipped'][ii].append(f_vv_flipped)
                non_overlapping_edges['dst_ei'][ii].append(ei)
                non_overlapping_edges['neighbor_edges'][ii].update(
                    n_edges.ravel().unique().tolist())
                ei_added = True
                break
        if not ei_added:
            non_overlapping_edges['ff_flipped'].append([f_vv_flipped])
            non_overlapping_edges['dst_ei'].append([ei])
            non_overlapping_edges['neighbor_edges'].append(
                set(n_edges.ravel().tolist()))

    for ii in range(len(non_overlapping_edges['neighbor_edges'])):
        if len(non_overlapping_edges['neighbor_edges'][ii]) == 0:
            break
        ff_flipped = non_overlapping_edges['ff_flipped'][ii]
        ff_flipped = torch.vstack(ff_flipped)
        # _, f_color_var_ret = color_variance(vertices[1:], ff_flipped, image)
        _, f_color_var_ret = color_variance_sampling(
            vertices[1:], ff_flipped, image)
        f_color_var_ret = f_color_var_ret.reshape(-1, 2)

        dst_ei = non_overlapping_edges['dst_ei'][ii]
        dst_ei = torch.tensor(dst_ei, device=faces.device)
        e_color_var_after[dst_ei] = f_color_var_ret.sum(axis=1)

    # loss_change = 2 + \
    #     neighbor_degrees.sum(dim=-1) - edge_degrees.sum(dim=-1)  # E
    loss_change = e_color_var_after - e_color_var
    color_loss_change = loss_change.clone()
    candidates = torch.logical_and(
        loss_change < -flip_loss_epsilon, edge_is_inside)  # E
    # candidates[boundary_neighboring_edges] = False
    loss_change = loss_change[candidates]  # E'
    if loss_change.shape[0] == 0:
        return 0, None

    edges_neighbors = torch.concat(
        (edges[candidates], neighbors[candidates]), dim=-1)  # E',4
    _, order = loss_change.sort(descending=True, stable=stable)  # E'
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0, len(rank), device=rank.device)
    vertex_rank = torch.zeros((V, 4), dtype=torch.long, device=device)  # V,4
    torch_scatter.scatter_max(
        src=rank[:, None].expand(-1, 4), index=edges_neighbors, dim=0, out=vertex_rank)
    vertex_rank, _ = vertex_rank.max(dim=-1)  # V
    neighborhood_rank, _ = vertex_rank[edges_neighbors].max(dim=-1)  # E'
    flip = rank == neighborhood_rank  # E'

    if with_normal_check:
        #  cl-<-----e1     e0,e1...edge, e0<e1
        #   |      /A      L,R....left and right face
        #   |  L /  |      both triangles ordered counter clockwise
        #   |  / R  |      normals pointing out of screen
        #   V/      |
        #   e0---->-cr
        v = vertices[edges_neighbors]  # E",4,3
        v = v - v[:, 0:1]  # make relative to e0
        e1 = v[:, 1]
        cl = v[:, 2]
        cr = v[:, 3]
        # sum of old normal vectors
        n = torch.cross(e1, cl) + torch.cross(cr, e1)
        flip.logical_and_(torch.sum(n*torch.cross(cr, cl),
                          dim=-1) > 0)  # first new face
        # second new face
        flip.logical_and_(torch.sum(n*torch.cross(cl-e1, cr-e1), dim=-1) > 0)

    flip_edges_neighbors = edges_neighbors[flip]  # E",4
    flip_edge_to_face = edge_to_face[candidates, :, 0][flip]  # E",2
    flip_faces = flip_edges_neighbors[:, [[0, 3, 2], [1, 2, 3]]]  # E",2,3
    faces.scatter_(dim=0, index=flip_edge_to_face.reshape(-1,
                   1).expand(-1, 3), src=flip_faces.reshape(-1, 3))

    if vis_itr >= 0:
        snapshot = Path('./local_snapshots/debug/')
        fig, ax = plt.subplots()
        plt.title(f'itr: {vis_itr}')
        png_name = snapshot / f'c_{vis_itr:05d}.svg'
        ax.imshow(image, alpha=0.8, extent=[
            0, image.width, image.height, 0])
        plot_VF(vertices[1::], faces[1::]-1, ax=ax,
                e=(edges[color_loss_change < 0]-1))
        plt.savefig(png_name, dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        plt.title(f'itr: {vis_itr}')
        png_name = snapshot / f'flip_{vis_itr:05d}.svg'
        ax.imshow(image, alpha=0.8, extent=[
            0, image.width, image.height, 0])
        plot_VF(vertices[1::], faces[1::]-1, ax=ax,
                e=(edges[color_loss_change < -flip_loss_epsilon]-1))
        plt.savefig(png_name, dpi=300)
        plt.close()

    return torch.sum(flip.int()), flip_edge_to_face.reshape(-1) - 1


def calc_flip_edges_combined(
        vertices: torch.Tensor,  # V,3 first unused
        faces: torch.Tensor,  # F,3 long, first must be 0, 0 for unused
        face_to_edge: torch.Tensor,  # F,3 long 0 for unused
        edges: torch.Tensor,  # E,2 long, first must be 0, 0 for unused, lower vertex index first
        edge_to_face: torch.Tensor,  # E,[left,right],[face,side]
        f_color_var: torch.Tensor,
        image: Image,
        with_normal_check: bool = True,  # check face normal flips
        stable: bool = False,  # only for unit testing
        harmonic_lambda: float = 0.1,
        vis_itr: int = -1,
):
    V = vertices.shape[0]
    E = edges.shape[0]
    device = vertices.device
    neighbor_corner = (edge_to_face[:, :, 1] + 2) % 3  # go from side to corner
    neighbors = faces[edge_to_face[:, :, 0], neighbor_corner]  # E,LR=2
    edge_is_inside = neighbors.all(dim=-1)  # E

    # Find the current harmonic energy. Define f as color norm
    image_array = np.array(image)
    if len(image_array.shape) == 2:
        image_array = image_array[..., np.newaxis]
    samples_pixel = vertices[1:].floor().int()
    samples_pixel[:, 0] = torch.clamp(
        samples_pixel[:, 0], 0, image.width - 1)
    samples_pixel[:, 1] = torch.clamp(
        samples_pixel[:, 1], 0, image.height - 1)
    samples_pixel = samples_pixel.detach().cpu().numpy()

    norm_vertices = vertices[1::] / \
        torch.tensor([image.width, image.height, 1], device=vertices.device)
    norm_vertices = norm_vertices.detach().cpu().numpy()
    cotm = -igl.cotmatrix(norm_vertices,
                          (faces[1::] - 1).detach().cpu().numpy())
    harmonic_energy = cotm.trace()

    # Find the current color variance of the two adjacent faces
    e_color_var = f_color_var[edge_to_face[:, :, 0]].sum(axis=1)

    # Calculate the color variance of the new faces
    orig_faces = faces[1:] - 1
    e_color_var_after = e_color_var.clone()
    # loss_epsilon = 1
    loss_epsilon = 0.5
    non_overlapping_edges = {
        'ff_flipped': [[]],
        'dst_ei': [[]],
        'neighbor_edges': [set()],
    }
    harm_energy_flipped = torch.zeros(E, dtype=vertices.dtype, device=device)
    harm_energy_before = torch.zeros(E, dtype=vertices.dtype, device=device)
    energe_checked = False
    vertices_norm = vertices[1::].clone()
    for ei in range(E):
        f_neighbors = edge_to_face[ei, :, 0] - 1
        f_vv = orig_faces[f_neighbors].clone()
        vertices_not_shared = []
        for vi in range(f_vv.size(0)):
            unique_vertex = [ii for ii, vertex in enumerate(f_vv[vi])
                             if vertex not in f_vv[1-vi]]
            if len(unique_vertex) == 0:
                continue
            vertices_not_shared.append(unique_vertex[0])
        if len(vertices_not_shared) < 2:
            continue

        if (not edge_is_inside[ei] or e_color_var[ei] < loss_epsilon):
            # if not edge_is_inside[ei]:
            continue

        f_vv_flipped = torch.tensor([[f_vv[0, vertices_not_shared[0]], f_vv[0, (vertices_not_shared[0] + 1) % 3], f_vv[1, vertices_not_shared[1]]],
                                     [f_vv[1, vertices_not_shared[1]], f_vv[1, (vertices_not_shared[1] + 1) % 3], f_vv[0, vertices_not_shared[0]]]], device=faces.device)

        # Compute current harmonic energy
        # Local version:
        v_neighbors = orig_faces[f_neighbors].ravel().unique()
        ring_faces = []
        for vi in v_neighbors:
            ring_faces.append(torch.nonzero((orig_faces == vi).any(1)))
        ring_faces = torch.cat(ring_faces).unique()
        local_neighbors = orig_faces[ring_faces].clone()

        def compute_local_harmonic_energy(ln):
            nv, nf, i, j = igl.remove_unreferenced(
                norm_vertices, ln.detach().cpu().numpy())
            cotm = -igl.cotmatrix(nv, nf)
            harm_energy = cotm.trace()

            return harm_energy

        he_before = compute_local_harmonic_energy(local_neighbors)
        harm_energy_before[ei] = float(he_before)

        local_neighbors[ring_faces == f_neighbors[0]] = f_vv_flipped[0]
        local_neighbors[ring_faces == f_neighbors[1]] = f_vv_flipped[1]

        he_after = compute_local_harmonic_energy(local_neighbors)
        harm_energy_flipped[ei] = float(he_after)

        # Full version:
        # if not energe_checked:
        if False:
            orig_faces[f_neighbors] = f_vv_flipped
            cotm = -igl.cotmatrix(vertices[1::].detach().cpu(
            ).numpy(), orig_faces.detach().cpu().numpy())
            # harm_energy = C.T @ cotm @ C
            harm_energy = cotm.trace()
            orig_faces[f_neighbors] = f_vv

            energy_diff = abs((harm_energy-harmonic_energy) -
                              float(he_after-he_before))
            print(f'\tLocal vs global harmonic energy: {energy_diff}')

            energe_checked = True

            # harm_energy_before[ei] = float(harmonic_energy)
            # harm_energy_flipped[ei] = float(harm_energy)

        # Prepare for color variance computation
        ei_added = False
        n_edges = face_to_edge[edge_to_face[ei, :, 0]]
        for ii in range(len(non_overlapping_edges['neighbor_edges'])):
            if ei not in non_overlapping_edges['neighbor_edges'][ii]:
                non_overlapping_edges['ff_flipped'][ii].append(f_vv_flipped)
                non_overlapping_edges['dst_ei'][ii].append(ei)
                non_overlapping_edges['neighbor_edges'][ii].update(
                    n_edges.ravel().unique().tolist())
                ei_added = True
                break
        if not ei_added:
            non_overlapping_edges['ff_flipped'].append([f_vv_flipped])
            non_overlapping_edges['dst_ei'].append([ei])
            non_overlapping_edges['neighbor_edges'].append(
                set(n_edges.ravel().tolist()))

    for ii in range(len(non_overlapping_edges['neighbor_edges'])):
        if len(non_overlapping_edges['neighbor_edges'][ii]) == 0:
            break
        ff_flipped = non_overlapping_edges['ff_flipped'][ii]
        ff_flipped = torch.vstack(ff_flipped)
        # _, f_color_var_ret = color_variance(vertices[1:], ff_flipped, image)
        _, f_color_var_ret = color_variance_sampling(
            vertices[1:], ff_flipped, image)
        assert isinstance(
            f_color_var_ret, torch.Tensor), 'No sample hits the mesh'
        dst_ei = non_overlapping_edges['dst_ei'][ii]
        dst_ei = torch.tensor(dst_ei, device=faces.device)
        f_color_var_ret = f_color_var_ret.reshape(-1, 2)
        e_color_var_after[dst_ei] = f_color_var_ret.sum(axis=1)

    # Normalize the two terms respectively
    # num_samples = image.width * image.height
    # num_vertices = vertices[1:].shape[0]
    num_samples = 1
    num_vertices = 1

    color_loss_change = e_color_var_after - e_color_var
    dirichlet_loss_change = harm_energy_flipped - harm_energy_before
    loss_change = (color_loss_change.double() / num_samples +
                   harmonic_lambda * dirichlet_loss_change.double() / num_vertices)
    candidates = torch.logical_and(loss_change < 0, edge_is_inside)  # E
    # candidates = torch.logical_and(
    #     color_loss_change < -loss_epsilon, candidates)  # E
    # candidates[boundary_neighboring_edges] = False

    loss_change = loss_change[candidates]  # E'
    if loss_change.shape[0] == 0:
        return 0, None

    edges_neighbors = torch.concat(
        (edges[candidates], neighbors[candidates]), dim=-1)  # E',4
    _, order = loss_change.sort(descending=True, stable=stable)  # E'
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0, len(rank), device=rank.device)
    vertex_rank = torch.zeros((V, 4), dtype=torch.long, device=device)  # V,4
    torch_scatter.scatter_max(
        src=rank[:, None].expand(-1, 4), index=edges_neighbors, dim=0, out=vertex_rank)
    vertex_rank, _ = vertex_rank.max(dim=-1)  # V
    neighborhood_rank, _ = vertex_rank[edges_neighbors].max(dim=-1)  # E'
    flip = rank == neighborhood_rank  # E'

    if with_normal_check:
        #  cl-<-----e1     e0,e1...edge, e0<e1
        #   |      /A      L,R....left and right face
        #   |  L /  |      both triangles ordered counter clockwise
        #   |  / R  |      normals pointing out of screen
        #   V/      |
        #   e0---->-cr
        v = vertices[edges_neighbors]  # E",4,3
        v = v - v[:, 0:1]  # make relative to e0
        e1 = v[:, 1]
        cl = v[:, 2]
        cr = v[:, 3]
        # sum of old normal vectors
        n = torch.cross(e1, cl) + torch.cross(cr, e1)
        flip.logical_and_(torch.sum(n*torch.cross(cr, cl),
                          dim=-1) > 0)  # first new face
        # second new face
        flip.logical_and_(torch.sum(n*torch.cross(cl-e1, cr-e1), dim=-1) > 0)

    flip_edges_neighbors = edges_neighbors[flip]  # E",4
    flip_edge_to_face = edge_to_face[candidates, :, 0][flip]  # E",2
    flip_faces = flip_edges_neighbors[:, [[0, 3, 2], [1, 2, 3]]]  # E",2,3
    faces.scatter_(dim=0, index=flip_edge_to_face.reshape(-1,
                   1).expand(-1, 3), src=flip_faces.reshape(-1, 3))

    return torch.sum(flip.int()), flip_edge_to_face.reshape(-1) - 1
