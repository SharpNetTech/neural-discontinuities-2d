from pathlib import Path
import os

import torch
import numpy as np

from tools.utils import temporary_directory

from neural.nerf2D_tri import MLPHybrid
from geometry.mesh_triangle import TriangleMesh


@torch.no_grad()
def precompute_unknown_discontinuous_feature_lookup(mlp, mlp_compressed):
    # Precompute the per vertex umbrella
    # 1. v -> {v_to}
    max_adj_e = 0
    v2v = []
    v2e = []
    boundary_e = set(tuple(sorted(e))
                     for loop in mlp.mesh.boundary_edges for e in loop)
    for vid in range(mlp.mesh.v.shape[0]):
        def find_adjacent_eid(vid):
            row_indices = torch.nonzero(
                torch.any(torch.eq(mlp.mesh_e, torch.tensor(vid)), dim=1), as_tuple=True)[0]
            assert len(row_indices) >= 1, f'Isolated vertex {vid}'
            return row_indices
        eids = find_adjacent_eid(vid)
        # Filter out continuous edges
        eids = [e for e in eids if not mlp.w_mask[e.item()]]
        max_adj_e = max(max_adj_e, len(eids))

        # Pick reference edges if the vertex is on the boundary
        boundary_to_vids = []
        for e in eids:
            ee = tuple(mlp.mesh_e[e.item()].tolist())
            if ee in boundary_e:
                boundary_to_vids.append(mlp.mesh_e[e.item(
                )][0] if mlp.mesh_e[e.item()][0] != vid else mlp.mesh_e[e.item()][1])
        picked_vid = None
        if len(boundary_to_vids):
            assert len(
                boundary_to_vids) > 1, f'Boundary vertex {vid} has only one adjacent edge'
            # Find the edge that is in a CW order (since Y is flipped)
            for vid_to in boundary_to_vids:
                mask = torch.any(mlp.mesh.f == vid, dim=1) & torch.any(
                    mlp.mesh.f == vid_to, dim=1)
                fid = torch.nonzero(mask, as_tuple=False).squeeze()
                assert len(fid.shape) == 0

                f_vidx = torch.nonzero(mlp.mesh.f[fid] == vid).squeeze()
                # CW triangle
                if mlp.mesh.f[fid, (f_vidx + 1) % 3] != vid_to:
                    picked_vid = int(vid_to)
                    break
            assert picked_vid is not None, f'Cannot find the CCW edge for boundary vertex {vid}'

        vv_list = [int(mlp.mesh_e[e.item()][0]) if mlp.mesh_e[e.item(
        )][0] != vid else int(mlp.mesh_e[e.item()][1]) for e in eids]
        v2e_local = {vv: int(eids[i].item())
                     for i, vv in enumerate(vv_list)}
        if picked_vid is not None:
            vv_list = [picked_vid] + \
                [vv for vv in vv_list if vv != picked_vid]
        vv = torch.tensor(vv_list, device=mlp.mesh.v.device)
        v2v.append(vv.int().flatten())

        # ee = torch.tensor([e.item() for e in eids])
        ee = torch.tensor([v2e_local[vv] for vv in vv_list])
        v2e.append(ee.int().reshape(-1, 1))
    mlp_compressed.v2v = torch.nn.utils.rnn.pad_sequence(
        v2v, batch_first=True, padding_value=-1)
    mlp_compressed.thetas = -1 * torch.ones_like(
        mlp_compressed.v2v, dtype=mlp_compressed.mesh.v.dtype, device=mlp_compressed.mesh.v.device)

    # Compute thetas
    mlp_compressed.update_discontinuity_thetas()

    # 2. (v) -> {(w, l, r)}
    # w is indexed by the edge index
    mlp_compressed.ve_features = -1 * torch.ones([mlp_compressed.mesh.v.shape[0], max_adj_e, 3],
                                                 dtype=mlp_compressed.mesh.f.dtype, device=mlp_compressed.mesh.v.device)

    return mlp_compressed


@torch.no_grad()
def compress_mlp(mlp):
    # Make an empty copy with the mesh info
    layer_dims = [mlp.fea_dim]
    layer_dims += [l.out_features for l in mlp.layers]

    V = mlp.mesh.get_v().cpu().numpy()
    F = mlp.mesh.f.cpu().numpy()
    mesh_compress = TriangleMesh(png=None, svg=None, inside_only=True)
    svg_info = (mlp.mesh.polys, mlp.mesh.attributes,
                (mlp.image.width, mlp.image.height))
    mesh_compress.build_mesh(svg_info, V, F)

    mlp_compressed = MLPHybrid(
        mlp.image, layer_dims, mesh_compress,
        feature_type=mlp.feature_type,
        opt_config=mlp.opt_config, snapshot_dir=mlp.snapshot_dir)

    # Compress the network
    # 1. The per-vertex biases are the same
    mlp_compressed.bias = torch.nn.Parameter(mlp.bias, requires_grad=True)

    # 2. Get the compact discontinuous edges
    w_count = torch.nonzero(mlp.w_mask).shape[0]
    mlp_compressed.w = torch.nn.Parameter(
        torch.zeros([w_count, 1]), requires_grad=True)
    mlp_compressed.w_mask = torch.ones([mlp_compressed.w.shape[0], 1]).bool()
    w2w = {}
    w2w2 = {}
    for i, w_idx in enumerate(torch.nonzero(mlp.w_mask)):
        w2w[i] = w_idx.item()
        w2w2[w_idx.item()] = i

    # mlp.features
    # mlp.v2v
    # mlp.thetas
    # mlp.ve_features

    # 3. Get the compact lookup info
    mlp_compressed = precompute_unknown_discontinuous_feature_lookup(
        mlp, mlp_compressed)

    # 4. Get the compact features
    ve_features_extended = mlp.ve_features[list(range(V.shape[0]))]
    mlp_w = mlp.w.detach().clone()
    mlp_w[~mlp.w_mask] = torch.nan

    # 5. Write the compressed ws and features
    for i in range(len(mlp_compressed.w)):
        mlp_compressed.w[i] = mlp.w[w2w[i]]

    seem_fea = set()
    save_fea = []
    fea2fea = {}
    for i in range(ve_features_extended.shape[0]):
        f_count = 0
        for k in range(ve_features_extended.shape[1]):
            if torch.isnan(mlp_w[ve_features_extended[i, k, 0]]):
                continue

            assert i in w2w2, f'Vertex {i} is not attached to any discontinuous edge'
            mlp_compressed.ve_features[i, f_count,
                                       0] = w2w2[ve_features_extended[i, k, 0]]
            for j in range(ve_features_extended.shape[2]):
                if j == 0:
                    continue
                if int(ve_features_extended[i, k, j]) < 0:
                    continue
                fea_idx = int(ve_features_extended[i, k, j])

                if int(ve_features_extended[i, k, j]) not in seem_fea:
                    fea_idx_comp = len(save_fea)
                else:
                    fea_idx_comp = fea2fea[fea_idx]

                mlp_compressed.ve_features[i, f_count, j] = fea_idx_comp

                if int(ve_features_extended[i, k, j]) in seem_fea:
                    continue

                seem_fea.add(fea_idx)
                fea2fea[fea_idx] = len(save_fea)
                save_fea.append(mlp.features[ve_features_extended[i, k, j]])

            f_count += 1

    mlp_compressed.features = torch.nn.Parameter(save_fea, requires_grad=True)

    # For reference:
    # w = mlp.w[ve_features_extended[..., 0]]
    # l_fea = mlp.features[ve_features_extended[..., 1]]
    # r_fea = mlp.features[ve_features_extended[..., 2]]
    # bias = mlp.bias[vid]


def check_mlp_size(mlp):
    # 1. Get mesh
    v = mlp.mesh.get_v()
    f = mlp.mesh.f

    # 2. Get features
    bias = mlp.bias

    ve_features_extended = torch.clamp(
        mlp.ve_features[list(range(v.shape[0]))], min=0)
    mlp_w = mlp.w.detach().clone()
    mlp_w[~mlp.w_mask] = torch.nan

    seem_fea = set()
    save_fea = []
    save_he = []
    for i in range(ve_features_extended.shape[0]):
        for k in range(ve_features_extended.shape[1]):
            if torch.isnan(mlp_w[ve_features_extended[i, k, 0]]):
                continue
            for j in range(ve_features_extended.shape[2]):
                if j == 0:
                    continue
                if int(ve_features_extended[i, k, j]) < 0:
                    continue
                if int(ve_features_extended[i, k, j]) in seem_fea:
                    continue
                seem_fea.add(int(ve_features_extended[i, k, j]))
                save_fea.append(mlp.features[ve_features_extended[i, k, j]])

                save_he.append(torch.tensor([ve_features_extended[i, k, 0]]))

    features = [v.to(torch.float32).detach().cpu().numpy(),
                f.to(torch.int32).detach().cpu().numpy(),
                mlp_w[mlp.w_mask].to(torch.float32).detach().cpu().numpy(),
                bias.to(torch.float32).detach().cpu().numpy(),
                torch.stack(save_fea, dim=0).to(
        torch.float32).detach().cpu().numpy(),
        torch.nonzero(mlp.w_mask).to(
        torch.int32).detach().cpu().numpy(),
        torch.stack(save_he, dim=0).to(
        torch.int32).detach().cpu().numpy()
    ]
    for l in mlp.layers:
        if hasattr(l, 'weight'):
            features.append(l.weight.detach().cpu().numpy())
            features.append(l.bias.detach().cpu().numpy())
    with temporary_directory() as tmp_dir:
        fea_file = tmp_dir / Path('mat.npz')
        np.savez(fea_file, *features)
        mlp_size = os.path.getsize(fea_file)

    return mlp_size
