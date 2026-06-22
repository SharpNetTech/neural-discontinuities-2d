import math
from pathlib import Path
import time
import gc

import igl
from gpytoolbox import ray_mesh_intersect
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_sum
import torch.nn as nn
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from tqdm import tqdm

from geometry.softras_fit import softras_render
from learning.nerf2d_data import normalize_image
from neural.nerf2D import MLP
from neural.utils import laplace, barycentric, gradient
from tools.plot_utils import plot_mesh, to_pil_image
from tools.param import cross_epsilon
from tools.utils import compose_image
from tools.geometry_utils import compute_angles_atan2
from learning.sampler import subpixel_sample


def render(mlp, x):
    # Note x is in the [0, 1]x[0, 1] coordinate
    v = mlp.get_v().detach()

    if x.shape[1] == 2:
        c = np.column_stack(
            (x.detach().cpu().numpy(), -1 * np.ones(x.shape[0])))
    else:
        c = x.cpu().detach().numpy()
        c[:, 2] = -1
    c[:, [0, 1]] = c[:, [1, 0]]
    c[:, 0] *= mlp.mesh.size[0]
    c[:, 1] *= mlp.mesh.size[1]

    d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
    v_np = v.cpu().detach().numpy()
    v_np[:, 2] = 0
    _, ids, ll = ray_mesh_intersect(
        c, d, v_np, mlp.mesh.f.cpu().detach().numpy())

    # Differentiable computation of barycentric
    points = x[:, [1, 0]]
    points[:, 0] = points[:, 0] * mlp.mesh.size[0]
    points[:, 1] = points[:, 1] * mlp.mesh.size[1]
    l = barycentric(points, v[mlp.mesh.f[ids]])
    cc = points
    if cc.shape[1] == 2:
        cc = torch.cat((cc, torch.zeros_like(cc[:, 0]).unsqueeze(-1)), dim=1)

    # Compute colors
    colors, target_rows = mlp.interpolate(ids, l, c, cc)

    return colors, target_rows


class MLPHybrid(MLP):
    def __init__(self, image, layer_dims, mesh,
                 opt_config,
                 feature_type='per_vertex',
                 snapshot_dir=Path('./')):
        super(MLPHybrid, self).__init__(image, layer_dims,
                                        opt_config=opt_config, encoding_type='none', L=-1)
        self.mesh = mesh
        if 'mesh_modify' in self.opt_config and self.opt_config['mesh_modify'] and feature_type == 'unknown_discontinuity':
            if 'largesteps_reparam_config' in self.opt_config:
                print('Registering mesh U as a parameter for large-steps')
                self.config_largesteps(
                    self.opt_config['largesteps_reparam_config'])
            else:
                print('Registering mesh V as a parameter')
                self.v0 = self.mesh.get_v().detach()
                self.mesh_v = torch.nn.Parameter(self.v0, requires_grad=True)
        else:
            self.v0 = self.mesh.get_v().detach()

        self.fea_dim = layer_dims[0]

        self.snapshot_dir = snapshot_dir

        self.feature_type = feature_type
        if self.feature_type == 'per_vertex':
            self.init_features_per_vertex()
        elif self.feature_type == 'per_edge':
            self.init_features_per_edge()
        elif self.feature_type == 'discontinuity':
            self.init_features_discontinuity()
        elif self.feature_type == 'unknown_discontinuity':
            self.init_features_unknown_discontinuity()
        else:
            assert False, 'Unknown feature type'

    # All continuous features

    def init_features_per_vertex(self):
        # For now, put one feature per vertex
        self.features = torch.nn.Parameter(torch.zeros(
            [self.mesh.v.shape[0], self.fea_dim]), requires_grad=True)
        nn.init.xavier_normal_(self.features)

        # Triangle feature lookup (to support discontinuous features in the future)
        self.f_features = torch.zeros_like(
            self.mesh.f, dtype=self.mesh.f.dtype)
        self.f_features = self.mesh.f

    def interpolate_features_barycentric(self, ids, l):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        if isinstance(l, torch.Tensor):
            l = l.detach().cpu().numpy()

        # Filter 'ids' for non-negative values
        valid_flags = ids >= 0
        valid_ids = ids[valid_flags]

        # Extract the relevant 'fid' values
        fid = self.f_features[valid_ids]
        fea = self.features[fid]

        # Compute the interpolation weights 'lamb' only for valid IDs
        lamb = l[valid_flags]

        # Compute the interpolated features using vectorized operations
        lamb = torch.tensor(lamb, device=self.device, dtype=torch.float32)
        f = (lamb[:, 0, None] * fea[:, 0, :]) + \
            (lamb[:, 1, None] * fea[:, 1, :]) + \
            (lamb[:, 2, None] * fea[:, 2, :])

        # Create the 'target_rows' list
        target_rows = np.where(valid_flags)[0].tolist()

        return f, target_rows

    # All continuous features end

    # All continuous features on edge
    def init_features_per_edge(self):
        # Build edges
        e = igl.edges(self.mesh.f.detach().cpu().numpy())
        self.mesh_e = torch.tensor(e, dtype=self.mesh.f.dtype)
        self.mesh_e, _ = torch.sort(self.mesh_e, dim=1)

        # Bias per vertex
        self.bias = torch.nn.Parameter(torch.zeros(
            [self.mesh.v.shape[0], self.fea_dim]), requires_grad=True)
        # nn.init.xavier_normal_(self.bias)

        max_adj_e = 0
        v2v = []
        for vid in range(self.mesh.v.shape[0]):
            def find_adjacent_eid(vid):
                row_indices = torch.nonzero(
                    torch.any(torch.eq(self.mesh_e, torch.tensor(vid)), dim=1), as_tuple=True)[0]
                assert len(row_indices) >= 1, f'Isolated vertex {vid}'
                return row_indices
            eids = find_adjacent_eid(vid)
            max_adj_e = max(max_adj_e, len(eids))

            vv_list = [int(self.mesh_e[e.item()][0])
                       if self.mesh_e[e.item()][0] != vid else int(self.mesh_e[e.item()][1]) for e in eids]
            vv = torch.tensor(vv_list, device=self.mesh.v.device)
            v2v.append(vv.int().flatten())

        self.v2v = torch.nn.utils.rnn.pad_sequence(
            v2v, batch_first=True, padding_value=-1)

        self.ve_features = -1 * torch.ones(
            [self.mesh.v.shape[0], max_adj_e], dtype=self.mesh.f.dtype, device=self.mesh.v.device)
        fea_idx = 0
        for vid in range(self.mesh.v.shape[0]):
            for j in range(v2v[vid].shape[0]):
                vid_to = v2v[vid][j]

                # Record the edge feature index
                self.ve_features[vid, j] = fea_idx
                fea_idx += 1

        # The features are two per vertex-edge pair: left and right
        self.features = torch.nn.Parameter(torch.zeros(
            [fea_idx, self.fea_dim]), requires_grad=True)
        nn.init.xavier_normal_(self.features)

        return

    def interpolate_features_edge_mat(self, ids, l, c):
        ids = torch.tensor(ids, dtype=self.mesh.f.dtype).to(self.device)
        l = torch.tensor(l, dtype=self.mesh.v.dtype).to(self.device)

        # Filter 'ids' for non-negative values
        with torch.no_grad():
            valid_flags = ids >= 0
            valid_ids = ids[valid_flags]

            cc = torch.from_numpy(c).to(self.device)[valid_flags]
            cc = cc.type(self.mesh.v.dtype)
            cc[:, 2] = 0

        v = self.get_v().detach()
        # Look up per vertex all adjacent edges
        with torch.no_grad():
            # Get unique vertices
            vid = self.mesh.f[valid_ids]
            valid_v = v[vid]

            # Look up the adjacent edges for each vertex within the faces
            vid_next = torch.roll(vid, shifts=-1, dims=1)
            vid_prev = torch.roll(vid, shifts=1, dims=1)

            ve_i_next = torch.nonzero(
                self.v2v[vid] == vid_next.unsqueeze(-1))[:, 2].view(vid_next.shape[0], -1)
            ve_i_prev = torch.nonzero(
                self.v2v[vid] == vid_prev.unsqueeze(-1))[:, 2].view(vid_next.shape[0], -1)

            assert (self.v2v[vid, ve_i_next] ==
                    vid_next).all(), "self.v2v indexed wrongly"
            assert (self.v2v[vid, ve_i_prev] ==
                    vid_prev).all(), "self.v2v indexed wrongly"

            # Convert sample points to theta coordinates
            # Compute the angle to interpolate the edge features
            dir_next = v[vid_next] - valid_v
            dir_prev = v[vid_prev] - valid_v
            dir_thetas = cc.unsqueeze(1) - valid_v

            # TODO: Handle the degenerate case of hitting a vertex exactly

            # Compute the rotation angles
            # angle0 = compute_angles(dir_prev, dir_thetas)
            # angle1 = compute_angles(dir_prev, dir_next)
            angle0 = compute_angles_atan2(dir_prev, dir_thetas)
            angle1 = compute_angles_atan2(dir_prev, dir_next)

            # |S| x |v| (3x|f|) x |theta|
            t = angle0 / angle1
            # assert torch.all(torch.logical_and(t >= 0 - cross_epsilon, t <= 1 + cross_epsilon)
            #                  ), "Not all t are in the range [0, 1]"
            t = torch.clamp(t, 0, 1)

        fea_next = self.features[self.ve_features[vid, ve_i_next]]
        fea_prev = self.features[self.ve_features[vid, ve_i_prev]]
        bias = self.bias[vid]

        # Blend features between the two edges
        fea_thetas = t.unsqueeze(-1) * fea_next + \
            (1 - t).unsqueeze(-1) * fea_prev
        fea = fea_thetas + bias

        # Compute the interpolation weights 'lamb' only for valid IDs
        lamb = l[valid_flags]

        # Compute the interpolated features using vectorized operations
        f = (lamb[:, 0].unsqueeze(1) * fea[:, 0, :]) + \
            (lamb[:, 1].unsqueeze(1) * fea[:, 1, :]) + \
            (lamb[:, 2].unsqueeze(1) * fea[:, 2, :])

        # Create the 'target_rows' list
        target_rows = torch.where(valid_flags)[0].tolist()

        return f, target_rows

    # All continuous features on edge end

    # Known discontinuities from the input
    def init_features_discontinuity(self):
        # Face features are CCW (the first |F| rows) and CW (the second |F| rows)
        # The orientation is based on the orientation of the face (aka the CCW and CW half edge)
        self.f_features = -1 * torch.ones(
            (self.mesh.f.shape[0] * 2, self.mesh.f.shape[1]), dtype=self.mesh.f.dtype)

        # Check if a face has discontinuitous edges
        fea_count = 0
        e_set = set(tuple(row) for row in self.mesh.e_discont)
        v_discont = set(self.mesh.e_discont.flatten().tolist())
        v_fea = {}
        for fid in range(self.mesh.f.shape[0]):
            for i in range(self.mesh.f.shape[1]):
                j = (i+1) % self.mesh.f.shape[1]
                e = (min(self.mesh.f[fid][i], self.mesh.f[fid][j]),
                     max(self.mesh.f[fid][i], self.mesh.f[fid][j]))
                e = (int(e[0]), int(e[1]))
                if e in e_set:
                    # For discontinuous edges, we need to store two features
                    # per oriented edge (separately for two adjacent triangles)
                    self.f_features[fid, i] = fea_count
                    fea_count += 1

                    self.f_features[fid + self.mesh.f.shape[0], j] = fea_count
                    fea_count += 1
                else:
                    # For continuous edges
                    # Check if this vertex is adjacent to any discontinuous edge
                    vid = int(self.mesh.f[fid][i])
                    if vid not in v_discont:
                        # Both CCW and CW point to the same feature
                        if vid not in v_fea:
                            self.f_features[fid, i] = fea_count
                            self.f_features[fid +
                                            self.mesh.f.shape[0], i] = fea_count
                            v_fea[vid] = fea_count
                            fea_count += 1
                        else:
                            self.f_features[fid, i] = v_fea[vid]
                            self.f_features[fid +
                                            self.mesh.f.shape[0], i] = v_fea[vid]

        # The features are tightly packed based on the discontinuity edges
        self.features = torch.nn.Parameter(torch.zeros(
            [fea_count, self.fea_dim]), requires_grad=True)
        nn.init.xavier_normal_(self.features)

        # Precompute the discontinuous feature lookup
        self.precompute_discontinuous_feature_lookup()

    def find_first_discontinuous_feature(self, fea_indices, fid, i, delta_i):
        vid = self.mesh.f[fid, i]
        v = self.get_v().detach()
        center_v = v[vid].detach()
        mask = (self.mesh.f == vid).any(dim=1)
        indices_all = torch.nonzero(mask, as_tuple=False).squeeze()
        indices = indices_all[indices_all != fid]
        adjacent_faces = self.mesh.f[indices]

        def iterate_he(fid_src, i_cur, d_i):
            vid_next = self.mesh.f[fid_src, (i_cur + 3 + d_i) % 3]
            next_mask = (adjacent_faces == vid_next).any(dim=1)
            next_indices_all = indices[next_mask]
            next_indices = next_indices_all[next_indices_all != fid_src]

            # No next face. This shouldn't happen since
            # we should assign object boundary to be discontinuous as well
            if len(next_indices) == 0:
                return -1, -1, -1

            next_fid = next_indices[0]
            next_f = self.mesh.f[next_fid]
            return next_fid, torch.nonzero(next_f == vid).squeeze(), torch.nonzero((next_f != vid) & (next_f != vid_next)).squeeze()

        fid_itr = fid
        i_itr = i
        fea_idx = -1
        fea_fid = -1
        while True:
            next_fid, i_itr, next_i = iterate_he(fid_itr, i_itr, delta_i)
            if next_fid < 0 or fea_indices[next_fid, i_itr] >= 0:
                if next_fid < 0:
                    ff = self.features[fea_indices[fid, i]]
                    fea_idx = fea_indices[fid, i]
                    fea_fid = fid
                    side_vid = self.mesh.f[fid, i]
                    d = v[side_vid].detach()
                    print(f'Warning: cannot find the next face for {fid}, {i}')
                else:
                    ff = self.features[fea_indices[next_fid, i_itr]]
                    fea_idx = fea_indices[next_fid, i_itr]
                    fea_fid = next_fid
                    side_vid = self.mesh.f[next_fid, next_i]
                    d = v[side_vid].detach()
                d = d - center_v
                break

            fid_itr = next_fid
            if next_fid == fid:
                assert False, 'Cannot find discontinuous edge'

        return ff, side_vid, d, fea_idx, fea_fid

    @torch.no_grad()
    def precompute_discontinuous_feature_lookup(self):
        # Get all faces with discontinuous edges
        features_CCW = self.f_features[0:self.mesh.f.shape[0], :]
        features_CW = self.f_features[self.mesh.f.shape[0]::, :]

        # These two matrices save indices to the discontinuous features
        # found by iterating half edges adjacent to each vertex
        f_features_ccw = -1 * torch.ones_like(features_CCW)
        f_features_cw = -1 * torch.ones_like(features_CW)
        f_dst_ccw = -1 * torch.ones_like(features_CCW)
        f_dst_cw = -1 * torch.ones_like(features_CW)

        ff_ccw = -1 * torch.ones_like(features_CCW)
        ff_cw = -1 * torch.ones_like(features_CW)

        def get_discontinuous_fea_indices(i, fea_indices_ccw, fea_indices_cw,
                                          dst_ccw, dst_cw,
                                          fid_ccw, fid_cw):
            for j in range(fea_indices_ccw.shape[0]):
                if features_CCW[j, i] >= 0 and features_CCW[j, i] == features_CW[j, i]:
                    continue

                # Rotate CCW. Hit CW features.
                if features_CW[j, i] < 0:
                    _, vid_dst_ccw, _, fea_idx_ccw, ffid_ccw = self.find_first_discontinuous_feature(
                        features_CW, j, i, 2)
                else:
                    fea_idx_ccw = features_CW[j, i]
                    vid_dst_ccw = self.mesh.f[j, (i + 2) % 3]
                    ffid_ccw = j

                # Rotate CW. Hit CCW features.
                if features_CCW[j, i] < 0:
                    _, vid_dst_cw, _, fea_idx_cw, ffid_cw = self.find_first_discontinuous_feature(
                        features_CCW, j, i, -2)
                else:
                    fea_idx_cw = features_CCW[j, i]
                    vid_dst_cw = self.mesh.f[j, (i + 3 - 2) % 3]
                    ffid_cw = j

                fea_indices_ccw[j, i] = fea_idx_ccw
                fea_indices_cw[j, i] = fea_idx_cw
                dst_ccw[j, i] = vid_dst_ccw
                dst_cw[j, i] = vid_dst_cw

                # For debugging
                fid_ccw[j, i] = ffid_ccw
                fid_cw[j, i] = ffid_cw

            return fea_indices_ccw.to(self.device), fea_indices_cw.to(self.device), \
                dst_ccw.to(self.device), dst_cw.to(self.device), \
                fid_ccw.to(self.device), fid_cw.to(self.device)

        f_features_ccw, f_features_cw, f_dst_ccw, f_dst_cw, ff_ccw, ff_cw = get_discontinuous_fea_indices(
            0, f_features_ccw, f_features_cw, f_dst_ccw, f_dst_cw, ff_ccw, ff_cw)
        f_features_ccw, f_features_cw, f_dst_ccw, f_dst_cw, ff_ccw, ff_cw = get_discontinuous_fea_indices(
            1, f_features_ccw, f_features_cw, f_dst_ccw, f_dst_cw, ff_ccw, ff_cw)
        f_features_ccw, f_features_cw, f_dst_ccw, f_dst_cw, ff_ccw, ff_cw = get_discontinuous_fea_indices(
            2, f_features_ccw, f_features_cw, f_dst_ccw, f_dst_cw, ff_ccw, ff_cw)

        self.f_features_ccw, self.f_features_cw, self.f_dst_ccw, self.f_dst_cw = \
            f_features_ccw, f_features_cw, f_dst_ccw, f_dst_cw

    def interpolate_features_discontinuity_mat(self, ids, l, c):
        ids = torch.tensor(ids, dtype=self.mesh.f.dtype).to(self.device)
        l = torch.tensor(l, dtype=self.mesh.v.dtype).to(self.device)

        # Filter 'ids' for non-negative values
        with torch.no_grad():
            valid_flags = ids >= 0
            valid_ids = ids[valid_flags]
            cc = torch.from_numpy(c).to(self.device)[valid_flags]
            cc = cc.type(self.mesh.v.dtype)
            cc[:, 2] = 0

            # Extract the relevant 'fid' values
            fid_CCW = self.f_features[valid_ids]
            fid_CW = self.f_features[valid_ids + self.mesh.f.shape[0]]

            # Narrow down valid_ids to only triangle indices indicating at least one discontinuous vertex
            is_cont_flag = (fid_CCW >= 0) & (fid_CCW == fid_CW)
            discont_row_ids = (~is_cont_flag).any(
                dim=1).nonzero(as_tuple=True)[0]
            valid_discont_ids = valid_ids[discont_row_ids]
            valid_c = cc[discont_row_ids]

        # Fetch the features of vertices not adjacent to any discontinuous edge
        fea_cont = torch.where(((fid_CCW >= 0) & (fid_CCW == fid_CW)).unsqueeze(2),
                               self.features[torch.clamp(fid_CCW, min=0)], torch.tensor(0.))

        # Interpolate the features of vertices adjacent to discontinuous edges
        features_ccw = self.features[torch.clamp(
            self.f_features_ccw[valid_discont_ids], min=0)]
        features_cw = self.features[torch.clamp(
            self.f_features_cw[valid_discont_ids], min=0)]

        # Compute the interpolation angles
        with torch.no_grad():
            v = self.get_v()

            # Select the vertices based on the intersection
            # |discontinuous F| x 3 (#V per F) x 3 (|D|)
            center_v = v[torch.clamp(self.mesh.f[valid_discont_ids], min=0)]
            dst_v_ccw = v[torch.clamp(
                self.f_dst_ccw[valid_discont_ids], min=0)]
            dst_v_cw = v[torch.clamp(self.f_dst_cw[valid_discont_ids], min=0)]

            # Compute direction vectors
            dir_c = valid_c.unsqueeze(1) - center_v
            dir_ccw = dst_v_ccw - center_v
            dir_cw = dst_v_cw - center_v

            # Handle the degenerated case of valid_c==center_v
            dir_c = torch.where(torch.linalg.norm(dir_c, dim=2).unsqueeze(
                2) > 0, dir_c, torch.tensor([0., 0., 1.]).to(self.device))

            # Compute the rotation angles
            # Calculate CCW angle
            dot_product_ccw = torch.sum(dir_c * dir_ccw, dim=2)
            norm_product_ccw = torch.linalg.norm(
                dir_c, dim=2) * torch.linalg.norm(dir_ccw, dim=2)
            cos_angle_ccw = torch.clamp(
                dot_product_ccw / norm_product_ccw, -1.0, 1.0)
            angle_ccw_init = torch.acos(cos_angle_ccw)
            cross_product_ccw = torch.linalg.cross(dir_ccw, dir_c, dim=2)
            angle_ccw_deg = torch.where(cross_product_ccw[:, :, 2] > 0,
                                        2 * torch.pi - angle_ccw_init, angle_ccw_init)

            # Handle the degeneracy for the discontinuous end
            angle_ccw = torch.where((self.f_dst_ccw[valid_discont_ids] == self.f_dst_cw[valid_discont_ids]) & (
                self.f_dst_ccw[valid_discont_ids] >= 0) & (cross_product_ccw[:, :, 2] == 0),
                2 * torch.pi - angle_ccw_init, angle_ccw_deg)

            # Calculate CW angle
            dot_product_cw = torch.sum(dir_c * dir_cw, dim=2)
            norm_product_cw = torch.linalg.norm(
                dir_c, dim=2) * torch.linalg.norm(dir_cw, dim=2)
            cos_angle_cw = torch.clamp(
                dot_product_cw / norm_product_cw, -1.0, 1.0)
            angle_cw_init = torch.acos(cos_angle_cw)
            cross_product_cw = torch.linalg.cross(dir_cw, dir_c, dim=2)
            angle_cw = torch.where(cross_product_cw[:, :, 2] < 0,
                                   2 * torch.pi - angle_cw_init, angle_cw_init)

        discont_fea = (angle_ccw/(angle_ccw+angle_cw)).unsqueeze(2) * \
            features_ccw + (angle_cw/(angle_ccw+angle_cw)
                            ).unsqueeze(2) * features_cw
        discont_fea_ext = torch.zeros_like(fea_cont)
        discont_fea_ext[discont_row_ids] = discont_fea

        fea = torch.where(is_cont_flag.unsqueeze(2), fea_cont, discont_fea_ext)

        # Compute the interpolation weights 'lamb' only for valid IDs
        lamb = l[valid_flags]

        # Compute the interpolated features using vectorized operations
        f = (lamb[:, 0].unsqueeze(1) * fea[:, 0, :]) + \
            (lamb[:, 1].unsqueeze(1) * fea[:, 1, :]) + \
            (lamb[:, 2].unsqueeze(1) * fea[:, 2, :])

        # Create the 'target_rows' list
        target_rows = torch.where(valid_flags)[0].tolist()

        # Visualize the triangle hits of the discontinuous edges
        if False:
            with torch.no_grad():
                for j in range(cc.shape[0]):
                    if cc[j, 0] != 255 or cc[j, 1] != 332 or valid_ids[j] < 0:
                        continue
                    debug_fid = valid_ids[j]
                    debug_f = self.mesh.f[debug_fid]

                    v = self.get_v()

                    # The hit triangle
                    tri_v = v[debug_f].cpu()
                    x, y = tri_v[:, 0], tri_v[:, 1]
                    ax = plot_mesh(self.mesh, discontinuity=True)
                    ax.fill(x, y, color='r', alpha=0.5, lw=0)

                    ax.scatter(cc[j, 0].cpu() + 0.5,
                               cc[j, 1].cpu() + 0.5, c='g', s=1)

                    plt.savefig(self.snapshot_dir /
                                f'hit_{cc[j, 1]}_{debug_fid}_left.png', dpi=400)
                    plt.close()

        return f, target_rows

    # Known discontinuities from the input end

    # Unknown discontinuities
    def init_features_unknown_discontinuity(self):
        # Build edges
        e = igl.edges(self.mesh.f.detach().cpu().numpy())
        self.mesh_e = torch.tensor(e, dtype=self.mesh.f.dtype)
        self.mesh_e, _ = torch.sort(self.mesh_e, dim=1)

        # Alpha values per edge
        if 'simple_fea' in self.opt_config and self.opt_config['simple_fea']:
            self.w = torch.nn.Parameter(
                torch.zeros([e.shape[0], self.fea_dim]), requires_grad=True)
        else:
            self.w = torch.nn.Parameter(
                torch.zeros([e.shape[0], 1]), requires_grad=True)
        nn.init.xavier_normal_(self.w)
        # self.w = torch.nn.Parameter(
        #     -10 * torch.ones([e.shape[0], 1]), requires_grad=True)
        self.w_mask = torch.ones([self.w.shape[0], 1]).bool()

        # Bias per vertex
        self.bias = torch.nn.Parameter(torch.zeros(
            [self.mesh.v.shape[0], self.fea_dim]), requires_grad=True)
        # nn.init.xavier_normal_(self.bias)

        # print('Precomputing for unknown discontinuity')
        # start_time = time.time()
        fea_count = self.precompute_unknown_discontinuous_feature_lookup()
        # end_time = time.time()
        # print(f'Precomputation time: {end_time - start_time}')

        # The features are two per vertex-edge pair: left and right
        self.features = torch.nn.Parameter(torch.zeros(
            [fea_count, self.fea_dim]), requires_grad=True)
        nn.init.xavier_normal_(self.features)

        return

    @torch.no_grad()
    def update_discontinuity_thetas(self):
        v = self.get_v().detach()

        v2v_extended = torch.clamp(self.v2v, min=0)
        dir_ref = (v[v2v_extended[:, 0]] - v).unsqueeze(1)
        dir_thetas = v[v2v_extended] - v.unsqueeze(1)

        # Compute the rotation angles
        # Calculate CCW angle
        angle_ccw = compute_angles_atan2(dir_ref, dir_thetas, set_ref=True)
        self.thetas = torch.where(self.v2v >= 0, angle_ccw, -1)

    @torch.no_grad()
    def precompute_unknown_discontinuous_feature_lookup(self, to_visualize=False):
        # Precompute the per vertex umbrella
        # 1. v -> {v_to}
        max_adj_e = 0
        v2v = []
        v2e = []
        boundary_e = set(tuple(sorted(e))
                         for loop in self.mesh.boundary_edges for e in loop)
        for vid in range(self.mesh.v.shape[0]):
            def find_adjacent_eid(vid):
                row_indices = torch.nonzero(
                    torch.any(torch.eq(self.mesh_e, torch.tensor(vid)), dim=1), as_tuple=True)[0]
                assert len(row_indices) >= 1, f'Isolated vertex {vid}'
                return row_indices
            eids = find_adjacent_eid(vid)
            max_adj_e = max(max_adj_e, len(eids))

            # Pick reference edges if the vertex is on the boundary
            boundary_to_vids = []
            for e in eids:
                ee = tuple(self.mesh_e[e.item()].tolist())
                if ee in boundary_e:
                    boundary_to_vids.append(self.mesh_e[e.item(
                    )][0] if self.mesh_e[e.item()][0] != vid else self.mesh_e[e.item()][1])
            picked_vid = None
            if len(boundary_to_vids):
                assert len(
                    boundary_to_vids) > 1, f'Boundary vertex {vid} has only one adjacent edge'
                # Find the edge that is in a CW order (since Y is flipped)
                for vid_to in boundary_to_vids:
                    mask = torch.any(self.mesh.f == vid, dim=1) & torch.any(
                        self.mesh.f == vid_to, dim=1)
                    fid = torch.nonzero(mask, as_tuple=False).squeeze()
                    assert len(fid.shape) == 0

                    f_vidx = torch.nonzero(self.mesh.f[fid] == vid).squeeze()
                    # CW triangle
                    if self.mesh.f[fid, (f_vidx + 1) % 3] != vid_to:
                        picked_vid = int(vid_to)
                        break
                assert picked_vid is not None, f'Cannot find the CCW edge for boundary vertex {vid}'

            vv_list = [int(self.mesh_e[e.item()][0]) if self.mesh_e[e.item(
            )][0] != vid else int(self.mesh_e[e.item()][1]) for e in eids]
            v2e_local = {vv: int(eids[i].item())
                         for i, vv in enumerate(vv_list)}
            if picked_vid is not None:
                vv_list = [picked_vid] + \
                    [vv for vv in vv_list if vv != picked_vid]
            vv = torch.tensor(vv_list, device=self.mesh.v.device)
            v2v.append(vv.int().flatten())

            # ee = torch.tensor([e.item() for e in eids])
            ee = torch.tensor([v2e_local[vv] for vv in vv_list])
            v2e.append(ee.int().reshape(-1, 1))
        self.v2v = torch.nn.utils.rnn.pad_sequence(
            v2v, batch_first=True, padding_value=-1)
        self.thetas = -1 * torch.ones_like(
            self.v2v, dtype=self.mesh.v.dtype, device=self.mesh.v.device)

        # Compute thetas
        self.update_discontinuity_thetas()

        # Visualize the theta
        if to_visualize:
            debug_vid = 0
            with torch.no_grad():
                v = self.get_v().detach().cpu().numpy()

                for vid in range(self.mesh.v.shape[0]):
                    # if vid != debug_vid:
                    #     continue

                    ax = plot_mesh(self.mesh, discontinuity=False)
                    ax.scatter(v[vid, 0], v[vid, 1], c='r', s=10)
                    for j in range(self.v2v[vid].shape[0]):
                        if self.v2v[vid][j] < 0:
                            continue
                        v_to = self.v2v[vid][j]
                        ax.plot([v[vid, 0], v[v_to, 0]],
                                [v[vid, 1], v[v_to, 1]], c='r', lw=1.5)

                        # Visualize the theta and eid
                        eid = v2e[vid][j]
                        vis_str = f'to: {v_to}; eid: {int(eid)}, {int(self.mesh_e[eid][0, 0])} - {int(self.mesh_e[eid][0, 1])};' + \
                            f'\ntheta: {self.thetas[vid, j]:.2f}'
                        ax.text((v[vid, 0] + v[v_to, 0])/2,
                                (v[vid, 1] + v[v_to, 1])/2, f'{vis_str}', fontsize=2)
                    plt.savefig(self.snapshot_dir / f'v2v_{vid}.png', dpi=400)
                    plt.close()

        # 2. (v) -> {(w, l, r)}
        # w is indexed by the edge index
        self.ve_features = -1 * torch.ones(
            [self.mesh.v.shape[0], max_adj_e, 3], dtype=self.mesh.f.dtype, device=self.mesh.v.device)

        fea_idx = 0
        for vid in range(self.mesh.v.shape[0]):
            for j in range(v2v[vid].shape[0]):
                vid_to = v2v[vid][j]

                # w is indexed by the edge index
                wid = v2e[vid][j]
                self.ve_features[vid, j, 0] = wid

                # Record the two left and right edges
                self.ve_features[vid, j, 1] = fea_idx
                fea_idx += 1

                self.ve_features[vid, j, 2] = fea_idx
                fea_idx += 1

        # Visualize the features
        if to_visualize:
            debug_vid = 0
            with torch.no_grad():
                v = self.get_v().detach().cpu().numpy()

                for vid in range(self.mesh.v.shape[0]):
                    # if vid != debug_vid:
                    #     continue

                    ax = plot_mesh(self.mesh, discontinuity=False)
                    ax.scatter(v[vid, 0], v[vid, 1], c='r', s=10)

                    for j in range(self.v2v[vid].shape[0]):
                        if self.v2v[vid][j] < 0:
                            continue

                        v_to = self.v2v[vid][j]

                        wid, lid, rid = self.ve_features[vid, j,
                                                         0], self.ve_features[vid, j, 1], self.ve_features[vid, j, 2]

                        ax.plot([v[vid, 0], v[v_to, 0]],
                                [v[vid, 1], v[v_to, 1]], c='r', lw=1.5)

                        # Visualize the theta and eid
                        vis_str = f'to: {v_to}; wid: {int(wid)};' + \
                            f'\nl: {int(lid)}; r: {int(rid)}'
                        ax.text((v[vid, 0] + v[v_to, 0])/2,
                                (v[vid, 1] + v[v_to, 1])/2, f'{vis_str}', fontsize=2)

                    plt.savefig(self.snapshot_dir / f'wlr_{vid}.png', dpi=400)
                    plt.close()

        return fea_idx

    @torch.no_grad()
    def round_w(self, threshold=0.1):
        # L1 of slopes
        ve_features_extended = torch.clamp(self.ve_features[:, ...], min=0)

        w = self.w[ve_features_extended[..., 0]]
        l_fea = self.features[ve_features_extended[..., 1]]
        r_fea = self.features[ve_features_extended[..., 2]]

        # Blend features per corner using its two adjacent corners and the corresponding alpha values
        w = self.get_w(w, self.w_mask[ve_features_extended[..., 0]])
        w_padded = torch.where(
            self.ve_features[..., 0].unsqueeze(-1) >= 0, w, 0)
        if 'simple_fea' in self.opt_config and self.opt_config['simple_fea']:
            slope = w_padded
        else:
            slope = w_padded * (l_fea - r_fea)
        slope_max = slope.abs().max(dim=-1).values

        eid_pack = self.ve_features[..., 0].ravel()
        slope_max = slope_max.ravel()
        slope_max = slope_max[eid_pack >= 0]
        eid_pack = eid_pack[eid_pack >= 0]

        e_max = torch.zeros(
            self.mesh_e.shape[0], dtype=torch.float32, device=w.device)
        scatter_max(src=slope_max, index=eid_pack.long(), dim=0, out=e_max)

        w_mask = e_max.unsqueeze(-1) > threshold

        # Update bias of now semi-continuous or continuous vertices
        v_continuous_edges = (~w_mask[ve_features_extended[..., 0]]).int()
        v_continuous_edges_count = torch.where(
            self.ve_features[..., 0].unsqueeze(-1) >= 0, v_continuous_edges, 0)
        # v_continuous = v_continuous_edges_count.squeeze(-1).sum(
        #     dim=-1) == (self.ve_features[..., 0] >= 0).sum(dim=-1)
        v_continuous = v_continuous_edges_count.squeeze(-1).sum(dim=-1) > 0
        vid = torch.nonzero(v_continuous, as_tuple=False).squeeze()

        v_offset = 1e-4 * torch.ones((1, 3), device=self.mesh.v.device)
        v_offset[:, 2] = 0

        v = self.get_v().detach()
        valid_v = v[vid]
        cc = valid_v + v_offset

        # Convert sample points to theta coordinates
        # Compute the angle wrt the reference edge
        # |S| x |v| (3x|f|) x MaxAdjE
        v2v_extended = torch.clamp(self.v2v[vid], min=0)
        # |S| x |v| (3x|f|) x 3
        dir_ref = v[v2v_extended[..., 0]] - valid_v
        dir_thetas = cc - valid_v

        # TODO: Handle the degenerate case of hitting a vertex exactly

        # Compute the rotation angles
        # Calculate CCW angle
        # angle_ccw = compute_angles(dir_ref, dir_thetas)
        angle_ccw = compute_angles_atan2(dir_ref, dir_thetas)
        # |S| x |v| (3x|f|) x |theta|
        thetas = self.thetas[vid]
        c_normalized = (angle_ccw.unsqueeze(-1) - thetas) / (2 * torch.pi)

        # |S| x |v| (3x|f|) x |theta|
        # Different thetas define different coordinate systems
        t = torch.fmod(c_normalized + 1, 1)

        ve_features_extended = torch.clamp(self.ve_features[vid], min=0)

        w = self.w[ve_features_extended[..., 0]]
        l_fea = self.features[ve_features_extended[..., 1]]
        r_fea = self.features[ve_features_extended[..., 2]]
        bias = self.bias[vid]

        # # Blend features per corner using its two adjacent corners and the corresponding alpha values
        w = self.get_w(w, self.w_mask[ve_features_extended[..., 0]])
        w_padded = torch.where(
            self.ve_features[vid][..., 0].unsqueeze(-1) >= 0, w, 0)
        fea_thetas = w_padded * r_fea
        # Or if we don't ignore the supposed-to-be small term w_padded * t.unsqueeze(-1)
        # fea_thetas = w_padded * r_fea + w_padded * t.unsqueeze(-1)
        # Or we shift to the mid point
        # fea_thetas = w_padded * r_fea + w_padded * (l_fea - r_fea) * 0.5

        continuous_edge_mask = (~w_mask[ve_features_extended[..., 0]]) & (
            self.ve_features[vid, ..., 0].unsqueeze(-1) >= 0)
        for i in range(fea_thetas.shape[2]):
            fea_thetas[..., i][~continuous_edge_mask.squeeze(-1)] = 0
        fea = fea_thetas.sum(1) + bias

        self.w[ve_features_extended[...,
                                    0].unsqueeze(-1)[continuous_edge_mask]] = -float('inf')
        self.bias[vid] = fea

        self.w_mask = torch.logical_and(self.w_mask, w_mask)

    @staticmethod
    def get_w(w, w_mask):
        # return torch.sigmoid(w / 1e-1)
        sig_w = torch.sigmoid(w / 1e-1)
        w_ret = torch.where(w_mask, sig_w, torch.zeros_like(w))

        return w_ret

    def config_largesteps(self, largesteps_reparam_config):
        self.largesteps_reparam_config = largesteps_reparam_config

        # Make sure v in mesh is updated
        self.mesh.v = self.mesh.v + self.mesh.delta_v
        del self.mesh.delta_v  # Delete the existing Parameter
        self.mesh.register_buffer('delta_v', torch.zeros_like(
            self.mesh.v, dtype=self.mesh.v.dtype))

        self.mesh_v = self.mesh.get_v().detach()
        self.v0 = self.mesh.get_v().detach()

        # Set up large-steps preconditioning
        self.mesh.M = compute_matrix(
            self.mesh_v, self.mesh.f.to(
                torch.int64), self.largesteps_reparam_config['lambda_reparam'],
            cotan=False if 'cotan' not in self.largesteps_reparam_config else self.largesteps_reparam_config[
                'cotan'])
        u = to_differential(self.mesh.M, self.mesh_v)
        self.mesh.u = torch.nn.Parameter(u, requires_grad=True)
        self.mesh_v = None
        self.get_v()

    def get_v(self):
        if hasattr(self, 'largesteps_reparam_config') and self.mesh.u is not None and self.mesh_v is None:
            if not self.mesh.M.is_coalesced():
                self.mesh.M = self.mesh.M.coalesce()
            self.mesh_v = from_differential(
                self.mesh.M, self.mesh.u, 'Cholesky')

            # This is for edge gradient debug visualization
            # self.mesh_v.requires_grad_(True)
            # self.mesh_v.retain_grad()

        if hasattr(self, 'mesh_v') and 'mesh_modify' in self.opt_config and self.opt_config['mesh_modify']:
            return self.mesh_v

        return self.mesh.get_v()

    def update_v(self):
        if hasattr(self, 'largesteps_reparam_config') and self.mesh.u is not None:
            if not self.mesh.M.is_coalesced():
                self.mesh.M = self.mesh.M.coalesce()
            self.mesh_v = from_differential(
                self.mesh.M, self.mesh.u, 'Cholesky')

            # This is for debug visualization
            self.mesh_v.requires_grad_(True)
            self.mesh_v.retain_grad()

        if 'mesh_modify' in self.opt_config and self.opt_config['mesh_modify']:
            self.mesh.v = self.mesh_v
            self.update_discontinuity_thetas()

    def interpolate_features_unknown_discontinuity_mat(self, ids, l, c, cc_=None):
        if isinstance(ids, np.ndarray):
            ids = torch.tensor(ids, dtype=self.mesh.f.dtype).to(self.device)
        if isinstance(l, np.ndarray):
            l = torch.tensor(l, dtype=self.mesh.v.dtype).to(self.device)

        # Filter 'ids' for non-negative values
        with torch.no_grad():
            valid_flags = ids >= 0
            valid_ids = ids[valid_flags]

            cc = torch.from_numpy(c).to(self.device)[valid_flags]
            cc = cc.type(self.mesh.v.dtype)
            cc[:, 2] = 0
        if isinstance(cc_, torch.Tensor):
            cc = cc_[valid_flags]

        v = self.get_v().detach()

        # Look up per vertex all adjacent edges
        with torch.no_grad():
            # if True:
            # Get unique vertices
            vid = self.mesh.f[valid_ids]
            valid_v = v[vid]

            # Convert sample points to theta coordinates
            # Compute the angle wrt the reference edge
            # |S| x |v| (3x|f|) x MaxAdjE
            v2v_extended = torch.clamp(self.v2v[vid], min=0)
            # |S| x |v| (3x|f|) x 3
            dir_ref = v[v2v_extended[..., 0]] - valid_v
            dir_thetas = cc.unsqueeze(1) - valid_v

            # TODO: Handle the degenerate case of hitting a vertex exactly

            # Compute the rotation angles
            # Calculate CCW angle
            # angle_ccw = compute_angles(dir_ref, dir_thetas)
            angle_ccw = compute_angles_atan2(dir_ref, dir_thetas)
            # |S| x |v| (3x|f|) x |theta|
            thetas = self.thetas[vid]
            c_normalized = (angle_ccw.unsqueeze(-1) - thetas) / (2 * torch.pi)

            # |S| x |v| (3x|f|) x |theta|
            # Different thetas define different coordinate systems
            t = torch.fmod(c_normalized + 1, 1)

            fea_dim = self.features.shape[-1]
            ve_features_extended = torch.clamp(self.ve_features[vid], min=0)
            l_indices_expanded = ve_features_extended[...,
                                                      1].view(-1, 1).long().expand(-1, fea_dim)
            r_indices_expanded = ve_features_extended[...,
                                                      2].view(-1, 1).long().expand(-1, fea_dim)

        # w = self.w[ve_features_extended[..., 0]]
        # l_fea = self.features[ve_features_extended[..., 1]]
        # r_fea = self.features[ve_features_extended[..., 2]]
        # bias = self.bias[vid]

        w = torch.gather(
            self.w, 0, ve_features_extended[..., 0].view(-1, 1).long()).view(
            ve_features_extended[..., 0].shape).unsqueeze(-1)
        l_fea = torch.gather(self.features, 0, l_indices_expanded).view(
            list(ve_features_extended[..., 1].shape)+[fea_dim])
        r_fea = torch.gather(self.features, 0, r_indices_expanded).view(
            list(ve_features_extended[..., 2].shape)+[fea_dim])
        bias = torch.gather(
            self.bias, 0, vid.view(-1, 1).long().expand(-1, fea_dim)).view(list(vid.shape)+[fea_dim])

        # # Blend features per corner using its two adjacent corners and the corresponding alpha values
        w = self.get_w(w, self.w_mask[ve_features_extended[..., 0]])
        w_padded = torch.where(
            self.ve_features[vid][..., 0].unsqueeze(-1) >= 0, w, 0)
        if 'simple_fea' in self.opt_config and self.opt_config['simple_fea']:
            fea_thetas = w_padded * t.unsqueeze(-1) + r_fea
        else:
            fea_thetas = (w_padded * (t.unsqueeze(-1) *
                                      l_fea + (1 - t).unsqueeze(-1) * r_fea))
        fea = fea_thetas.sum(2) + bias
        # fea = bias
        # fea = fea_thetas.sum(2)

        # # Compute the interpolation weights 'lamb' only for valid IDs
        # lamb = l[valid_flags]

        # # Compute the interpolated features using vectorized operations
        # f = (lamb[:, 0].unsqueeze(1) * fea[:, 0, :]) + \
        #     (lamb[:, 1].unsqueeze(1) * fea[:, 1, :]) + \
        #     (lamb[:, 2].unsqueeze(1) * fea[:, 2, :])

        # Compute the interpolation weights 'lamb' only for valid IDs
        lamb = l[valid_flags].unsqueeze(-1)

        # Compute the interpolated features using vectorized operations
        f = (lamb * fea).sum(dim=1)

        # Create the 'target_rows' list
        target_rows = torch.where(valid_flags)[0].tolist()

        return f, target_rows

    def interpolate_features_unknown_discontinuity_mat2(self, ids, l, c, cc_=None):
        if isinstance(ids, np.ndarray):
            ids = torch.tensor(ids, dtype=self.mesh.f.dtype).to(self.device)
        if isinstance(l, np.ndarray):
            l = torch.tensor(l, dtype=self.mesh.v.dtype).to(self.device)

        # Filter 'ids' for non-negative values
        with torch.no_grad():
            valid_flags = ids >= 0
            valid_ids = ids[valid_flags]

            cc = torch.from_numpy(c).to(self.device)[valid_flags]
            cc = cc.type(self.mesh.v.dtype)
            cc[:, 2] = 0
        if isinstance(cc_, torch.Tensor):
            cc = cc_[valid_flags]

        v = self.get_v().detach()

        v_fea = self.bias

        # Update the features of the vertices with adjacent discontinuous edges
        with torch.no_grad():
            # if True:
            # Get unique vertices
            f_vid = self.mesh.f[valid_ids]

            vid = torch.nonzero((self.ve_features[:, :, 0]+1).any(dim=1))
            valid_v = v[vid]

            # Convert sample points to theta coordinates
            # Compute the angle wrt the reference edge
            # |S| x |v| (3x|f|) x MaxAdjE
            v2v_extended = torch.clamp(self.v2v[vid], min=0)
            # |S| x |v| (3x|f|) x 3
            dir_ref = v[v2v_extended[..., 0]] - valid_v
            dir_thetas = cc.unsqueeze(1) - valid_v

            # TODO: Handle the degenerate case of hitting a vertex exactly

            # Compute the rotation angles
            # Calculate CCW angle
            # angle_ccw = compute_angles(dir_ref, dir_thetas)
            angle_ccw = compute_angles_atan2(dir_ref, dir_thetas)
            # |S| x |v| (3x|f|) x |theta|
            thetas = self.thetas[vid]
            c_normalized = (angle_ccw.unsqueeze(-1) - thetas) / (2 * torch.pi)

            # |S| x |v| (3x|f|) x |theta|
            # Different thetas define different coordinate systems
            t = torch.fmod(c_normalized + 1, 1)

            fea_dim = self.features.shape[-1]
            ve_features_extended = torch.clamp(self.ve_features[vid], min=0)
            l_indices_expanded = ve_features_extended[vid, :,
                                                      1].view(-1, 1).long().expand(-1, fea_dim)
            r_indices_expanded = ve_features_extended[vid, :,
                                                      2].view(-1, 1).long().expand(-1, fea_dim)

        # w = self.w[ve_features_extended[..., 0]]
        # l_fea = self.features[ve_features_extended[..., 1]]
        # r_fea = self.features[ve_features_extended[..., 2]]
        # bias = self.bias[vid]

        w = torch.gather(
            self.w, 0, ve_features_extended[vid, :, 0].view(-1, 1).long()).view(
            ve_features_extended[vid, :, 0].shape).unsqueeze(-1)
        l_fea = torch.gather(self.features, 0, l_indices_expanded).view(
            list(ve_features_extended[vid, :, 1].shape)+[fea_dim])
        r_fea = torch.gather(self.features, 0, r_indices_expanded).view(
            list(ve_features_extended[vid, :, 2].shape)+[fea_dim])
        bias = torch.gather(
            self.bias, 0, vid.view(-1, 1).long().expand(-1, fea_dim)).view(list(vid.shape)+[fea_dim])

        # # Blend features per corner using its two adjacent corners and the corresponding alpha values
        w = self.get_w(w, self.w_mask[ve_features_extended[vid, :, 0]])
        w_padded = torch.where(
            self.ve_features[vid][vid, :, 0].unsqueeze(-1) >= 0, w, 0)
        if 'simple_fea' in self.opt_config and self.opt_config['simple_fea']:
            fea_thetas = w_padded * t.unsqueeze(-1) + r_fea
        else:
            fea_thetas = (w_padded * (t.unsqueeze(-1) *
                                      l_fea + (1 - t).unsqueeze(-1) * r_fea))
        v_fea[vid] = fea_thetas.sum(2) + bias[vid]

        fea = v_fea[f_vid]
        # fea = bias
        # fea = fea_thetas.sum(2)

        # # Compute the interpolation weights 'lamb' only for valid IDs
        # lamb = l[valid_flags]

        # # Compute the interpolated features using vectorized operations
        # f = (lamb[:, 0].unsqueeze(1) * fea[:, 0, :]) + \
        #     (lamb[:, 1].unsqueeze(1) * fea[:, 1, :]) + \
        #     (lamb[:, 2].unsqueeze(1) * fea[:, 2, :])

        # Compute the interpolation weights 'lamb' only for valid IDs
        lamb = l[valid_flags].unsqueeze(-1)

        # Compute the interpolated features using vectorized operations
        f = (lamb * fea).sum(dim=1)

        # Create the 'target_rows' list
        target_rows = torch.where(valid_flags)[0].tolist()

        return f, target_rows

    # Unknown discontinuities end

    @torch.no_grad()
    def compute_Q(self):
        F = self.features
        Q = super(MLPHybrid, self).forward(F)

        return Q

    def interpolate(self, ids, l, c, cc):
        if hasattr(self, 'f_features') and (self.f_features.device != self.device):
            self.f_features = self.f_features.to(self.device)
        if hasattr(self, 'f_features_ccw') and (self.f_features_ccw.device != self.device):
            self.f_features_ccw, self.f_features_cw, self.f_dst_ccw, self.f_dst_cw = \
                self.f_features_ccw.to(self.device), self.f_features_cw.to(
                    self.device), self.f_dst_ccw.to(self.device), self.f_dst_cw.to(self.device)
        if hasattr(self, 'bias') and (self.bias.device != self.device):
            self.bias = self.bias.to(self.device)
        if hasattr(self, 'w_mask') and (self.w_mask.device != self.device):
            self.w_mask = self.w_mask.to(self.device)

        # Interpolate the features
        if self.feature_type == 'per_vertex':
            f, target_rows = self.interpolate_features_barycentric(ids, l)
        elif self.feature_type == 'per_edge':
            f, target_rows = self.interpolate_features_edge_mat(ids, l, c)
        elif self.feature_type == 'discontinuity':
            f, target_rows = self.interpolate_features_discontinuity_mat(
                ids, l, c)
        elif self.feature_type == 'unknown_discontinuity':
            f, target_rows = self.interpolate_features_unknown_discontinuity_mat(
                ids, l, c, cc_=cc)
        else:
            assert False, 'Unknown feature type'

        # Call MLP
        colors = super(MLPHybrid, self).forward(f)

        return colors, target_rows

    def forward(self, x):
        # Look up features for interpolation
        # Reference: x contains [float(y) / height, float(x) / width]
        c = np.column_stack(
            (x.detach().cpu().numpy(), -1 * np.ones(x.shape[0])))
        c[:, [0, 1]] = c[:, [1, 0]]
        c[:, 0] *= self.mesh.size[0]
        c[:, 1] *= self.mesh.size[1]

        # # Offset the center
        # c += np.array([0.5, 0.5, 0])

        d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
        # Disable vertex modification
        v = self.get_v().detach()
        v_np = v.cpu().detach().numpy()
        v_np[:, 2] = 0
        _, ids, ll = ray_mesh_intersect(
            c, d, v_np, self.mesh.f.cpu().detach().numpy())

        # Differentiable computation of barycentric
        points = x[:, [1, 0]]
        points[:, 0] = points[:, 0] * self.mesh.size[0]
        points[:, 1] = points[:, 1] * self.mesh.size[1]
        l = barycentric(points, v[self.mesh.f[ids]])
        cc = torch.hstack([points, torch.zeros(
            [points.shape[0], 1], device=points.device)])

        # Interpolate the features and evaluate
        colors, target_rows = self.interpolate(ids, l, c, cc)

        # Put color back (the positions outside the mesh are assigned white color)
        canvas = torch.ones([x.shape[0], self.layers[-1].out_features])
        canvas = canvas.to(self.device)
        canvas[target_rows] = colors

        return canvas

    def mlp(self, f):
        colors = super(MLPHybrid, self).forward(f)

        return colors

    def l1_loss(self, e_len_=None):
        loss_l1 = torch.nn.L1Loss()

        # L1 of w
        # w = self.get_w(self.w, self.w_mask)
        # loss += self.opt_config['l1_weight'] * \
        #     loss_l1(w, torch.zeros_like(w))

        # L1 of slopes
        ve_features_extended = torch.clamp(self.ve_features[:, ...], min=0)

        w = self.w[ve_features_extended[..., 0]]
        l_fea = self.features[ve_features_extended[..., 1]]
        r_fea = self.features[ve_features_extended[..., 2]]

        # Compute edge length
        v = self.get_v().detach()

        if not isinstance(e_len_, torch.Tensor):
            e = self.mesh_e.cuda()[ve_features_extended[..., 0]]
            e_len = torch.norm(v[e[..., 0]] - v[e[..., 1]], p=2, dim=-1)
        else:
            e_len = e_len_
        e_len_padded = torch.where(
            self.ve_features[..., 0].unsqueeze(-1) >= 0, e_len.unsqueeze(-1), -1)
        e_len_padded2 = torch.where(
            self.ve_features[..., 0].unsqueeze(-1) >= 0, e_len.unsqueeze(-1), 0)
        e_len = e_len_padded[e_len_padded >= 0]
        e_len_norm = e_len_padded2 / e_len.mean()

        # Blend features per corner using its two adjacent corners and the corresponding alpha values
        w = self.get_w(w, self.w_mask[ve_features_extended[..., 0]])
        w_padded = torch.where(
            self.ve_features[..., 0].unsqueeze(-1) >= 0, w, 0)
        if 'simple_fea' in self.opt_config and self.opt_config['simple_fea']:
            slope = w_padded * e_len_norm
        else:
            slope = w_padded * (l_fea - r_fea) * e_len_norm

        return loss_l1(slope, torch.zeros_like(slope))

    def boundary_loss(self):
        v = self.get_v()
        loss_l2_sq = torch.nn.MSELoss()
        boundary_loss = loss_l2_sq(
            v[self.mesh.boundary_vid], self.v0[self.mesh.boundary_vid])

        return boundary_loss

    @staticmethod
    def smoothness_loss(x, y_hat, output_norm=False):
        grad = gradient(y_hat, x)
        # dirichlet energy = \int |\nabla u|^2 dx
        grad_norm = (grad**2).sum(dim=-1)
        if output_norm:
            return grad_norm

        dirichlet_term = (grad_norm).mean()
        # laplace_term = (laplace(y_hat, x)**2).mean()

        return dirichlet_term

    def training_step(self, batch, batch_idx, to_mesh_modify=False):
        x_, y = batch

        # Turn grad on for autograd for the smoothness term
        x = x_.requires_grad_(True)

        if to_mesh_modify and 'mesh_modify' in self.opt_config and self.opt_config['mesh_modify']:
            assert 'sigma' in self.opt_config, 'sigma is not provided for mesh_modify'
            y_hat = softras_render(
                self, x, self.opt_config['sigma'], to_normalize=False)
        else:
            y_hat = self(x)

        # print(y_hat.shape, y.shape)
        # exit()

        loss_func = nn.MSELoss()
        loss = loss_func(y_hat, y)

        if (self.feature_type == 'unknown_discontinuity') and ('l1_weight' in self.opt_config):
            loss += self.opt_config['l1_weight'] * self.l1_loss()

        if (self.feature_type == 'unknown_discontinuity') and ('smooth_weight' in self.opt_config) and \
                (not hasattr(self, 'rendering')):
            loss += self.opt_config['smooth_weight'] * \
                self.smoothness_loss(x, y_hat)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output_dim = self.layers[-1].out_features
        assert output_dim == 3, 'Not fitting to the image'

        # Run the validation on the entire image since we are overfitting
        width, height = self.image.size
        y_coords, x_coords = torch.meshgrid(torch.arange(
            height), torch.arange(width), indexing='ij')
        y_coords = (y_coords.float() + 0.5) / height
        x_coords = (x_coords.float() + 0.5) / width
        coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)
        coords = coords.to(self.device)
        coords.requires_grad = False
        colors_mlp = self(coords)

        image_norm = normalize_image(self.image)
        width, height = self.image.size
        colors = []
        for y in range(height):
            for x in range(width):
                r, g, b = image_norm[y, x]
                colors.append(torch.tensor([r, g, b]))
        colors = torch.vstack(colors).to(self.device)

        loss_func = nn.MSELoss()
        loss = loss_func(colors_mlp, colors)
        # print('val_loss', loss)
        self.log('val_loss', loss)
        # exit()
        # self.log('val_loss', loss, sync_dist=True)

        return loss

    def mask(self, zoom=1):
        width, height = self.image.size
        width, height = int(width * zoom), int(height * zoom)
        # rot_angle = 45.0
        # for y in range(height):
        #     for x in range(width):
        #         coord = torch.tensor([float(y) / height, float(x) / width])
        #         coords.append(coord)
        # x = torch.vstack(coords)
        y_coords, x_coords = torch.meshgrid(torch.arange(
            height), torch.arange(width), indexing='ij')
        y_coords = y_coords.float() / height
        x_coords = x_coords.float() / width
        x = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)

        c = np.column_stack((x.cpu().numpy(), -1 * np.ones(x.shape[0])))
        c[:, [0, 1]] = c[:, [1, 0]]
        c[:, 0] *= self.mesh.size[0]
        c[:, 1] *= self.mesh.size[1]

        # Offset the center
        c += np.array([0.5, 0.5, 0])

        d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
        v = self.get_v().detach()
        v_np = v.cpu().detach().numpy()
        v_np[:, 2] = 0
        _, ids, _ = ray_mesh_intersect(
            c, d, v_np, self.mesh.f.cpu().detach().numpy())

        valid_flags = ids >= 0
        target_rows = np.where(valid_flags)[0].tolist()

        canvas = torch.zeros(
            [x.shape[0], self.layers[-1].out_features], device='cpu')
        canvas[target_rows] = torch.tensor(
            [1] * self.layers[-1].out_features, dtype=canvas.dtype)

        output_dim = self.layers[-1].out_features
        canvas = canvas.reshape(height, width, output_dim)

        image = to_pil_image(canvas)

        return image

    def evaluate_raw_soft(self, zoom=1.0, viewbox=None, x_=None):
        width, height = self.image.size
        width, height = int(width * zoom), int(height * zoom)
        output_w, output_h = width, height

        if isinstance(x_, torch.Tensor):
            coords = x_
        else:
            coords = []
            if viewbox:
                x, y, x2, y2 = viewbox
                w = x2 - x
                h = y2 - y

                x, y, w, h = x * zoom, y * zoom, w * zoom, h * zoom
                # print(f'x: {x}, y: {y}, w: {w}, h: {h}')
                if w > h:
                    output_w = int(width)
                    output_h = int(math.ceil(h * width / w))
                else:
                    output_h = int(height)
                    output_w = int(math.ceil(w * height / h))
                # print(f'output_w: {output_w}, output_h: {output_h}')
                y_coords, x_coords = torch.meshgrid(torch.linspace(
                    y, y + h, output_h), torch.linspace(x, x + w, output_w), indexing='ij')
                zoom_x = output_w / w
                zoom_y = output_h / h
                y_coords = (y_coords.float() + 0.5 / zoom_x)
                x_coords = (x_coords.float() + 0.5 / zoom_y)
            else:
                y_coords, x_coords = torch.meshgrid(torch.arange(
                    height), torch.arange(width), indexing='ij')
                y_coords = (y_coords.float() + 0.5)
                x_coords = (x_coords.float() + 0.5)

            y_coords = y_coords / height
            x_coords = x_coords / width

            coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)

        encoded_coord = coords.to(self.device)
        encoded_coord.requires_grad = False
        raw_pixel = softras_render(
            self, encoded_coord, self.opt_config['sigma'], to_normalize=False)

        return raw_pixel

    @torch.no_grad()
    def evaluate_spp(self, spp=1, zoom=1.0):
        width, height = self.image.size
        width, height = int(width * zoom), int(height * zoom)

        output_dim = self.layers[-1].out_features
        rendering_all = torch.zeros([width * height, output_dim],
                                    dtype=torch.float32, device=self.mesh.device)
        # dtype=torch.float16, device=self.mesh.device)
        int_spp_all = torch.zeros([width * height],
                                  #   dtype=self.mesh.f.dtype, device=self.mesh.device)
                                  dtype=torch.int16, device=self.mesh.device)

        rendering = torch.zeros([width * height, output_dim],
                                dtype=torch.float32, device=self.mesh.device)
        # dtype=torch.float16, device=self.mesh.device)
        int_spp = torch.zeros([width * height],
                              #   dtype=self.mesh.f.dtype, device=self.mesh.device)
                              dtype=torch.int16, device=self.mesh.device)

        num_patches = int(zoom)
        for i in range(num_patches):
            for j in range(num_patches):
                sample_range = [j * width // num_patches, min((j + 1) * width // num_patches, width),
                                i * height // num_patches, min((i + 1) * height // num_patches, height)]
                gc.collect()
                torch.cuda.empty_cache()

                coords = subpixel_sample(
                    width, height, spp=spp, sample_range=sample_range).to(self.device)
                encoded_coord = coords
                encoded_coord.requires_grad = False

                batch_size = 3 * 4 * 512 * 512
                # batch_size = int(512 * 512 / 2)
                for coord_batch in tqdm(encoded_coord.split(batch_size)):
                    int_colors = self(coord_batch)

                    # Average the pixel values
                    rendering.zero_()

                    samples_img = coord_batch * \
                        torch.tensor([height,
                                      width], device=self.mesh.device)
                    samples_pixel = torch.floor(samples_img).int()
                    samples_pixel[:, 0] = torch.clamp(
                        samples_pixel[:, 0], 0, height - 1)
                    samples_pixel[:, 1] = torch.clamp(
                        samples_pixel[:, 1], 0, width - 1)

                    for d in range(int_colors.shape[-1]):
                        scatter_sum(
                            src=int_colors[..., d].ravel().to(rendering.dtype),
                            index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., d])

                    # scatter_sum(
                    #     src=int_colors[..., 0].ravel().to(rendering.dtype),
                    #     index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 0])
                    # scatter_sum(
                    #     src=int_colors[..., 1].ravel().to(rendering.dtype),
                    #     index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 1])
                    # scatter_sum(
                    #     src=int_colors[..., 2].ravel().to(rendering.dtype),
                    #     index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 2])
                    rendering_all = rendering_all + rendering

                    int_spp.zero_()
                    int_counts = torch.ones_like(
                        int_colors[..., 0]).to(int_spp.dtype)
                    scatter_sum(
                        src=int_counts.ravel(),
                        index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=int_spp)
                    int_spp_all = int_spp_all + int_spp

        rendering_all = rendering_all / \
            torch.clamp(int_spp_all, min=1).unsqueeze(-1)

        raw_px = rendering_all.reshape(height, width, output_dim)
        raw_px = raw_px.cpu()

        # Alpha blending
        if output_dim == 4:
            raw_px = compose_image(self.image_norm, raw_px)

        image = to_pil_image(raw_px)

        return image

    @torch.no_grad()
    def evaluate_spp_old(self, spp=1, zoom=1.0):
        width, height = self.image.size
        width, height = int(width * zoom), int(height * zoom)
        coords = subpixel_sample(width, height, spp=spp).to(self.device)
        encoded_coord = coords
        encoded_coord.requires_grad = False

        output_dim = self.layers[-1].out_features
        rendering_all = torch.zeros([width * height, output_dim],
                                    dtype=torch.float32, device=self.mesh.device)
        int_spp_all = torch.zeros([width * height],
                                  dtype=self.mesh.f.dtype, device=self.mesh.device)

        rendering = torch.zeros([width * height, output_dim],
                                dtype=torch.float32, device=self.mesh.device)
        int_spp = torch.zeros([width * height],
                              dtype=self.mesh.f.dtype, device=self.mesh.device)

        batch_size = 3 * 4 * 512 * 512
        for coord_batch in tqdm(encoded_coord.split(batch_size)):
            int_colors = self(coord_batch)

            # Average the pixel values
            rendering.zero_()

            samples_img = coord_batch * \
                torch.tensor([height,
                              width], device=self.mesh.device)
            samples_pixel = torch.floor(samples_img).int()
            samples_pixel[:, 0] = torch.clamp(
                samples_pixel[:, 0], 0, height - 1)
            samples_pixel[:, 1] = torch.clamp(
                samples_pixel[:, 1], 0, width - 1)

            for d in range(int_colors.shape[-1]):
                scatter_sum(
                    src=int_colors[..., d].ravel(),
                    index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., d])
            # scatter_sum(
            #     src=int_colors[..., 0].ravel(),
            #     index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 0])
            # scatter_sum(
            #     src=int_colors[..., 1].ravel(),
            #     index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 1])
            # scatter_sum(
            #     src=int_colors[..., 2].ravel(),
            #     index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 2])
            rendering_all = rendering_all + rendering

            int_spp.zero_()
            int_counts = torch.ones_like(
                int_colors[..., 0]).to(self.mesh.f.dtype)
            scatter_sum(
                src=int_counts.ravel(),
                index=(samples_pixel[:, 0] * width + samples_pixel[:, 1]).long().ravel(), out=int_spp)
            int_spp_all = int_spp_all + int_spp

        rendering_all = rendering_all / \
            torch.clamp(int_spp_all, min=1).unsqueeze(-1)

        raw_px = rendering_all.reshape(height, width, output_dim)
        raw_px = raw_px.cpu()

        # Alpha blending
        if output_dim == 4:
            raw_px = compose_image(self.image_norm, raw_px)

        image = to_pil_image(raw_px)

        return image

    @torch.no_grad()
    def for_sharpnet(self, roi=(0,1,0,1), resolution=512, dim=1):

        X = torch.linspace(roi[0], roi[1], resolution)
        Y = torch.linspace(roi[2], roi[3], resolution)

        yy, xx = torch.meshgrid(Y, X, indexing="ij")
        yy = yy.reshape(-1, 1)
        xx = xx.reshape(-1, 1)
        coords = torch.concatenate([yy, xx], dim=-1).to(self.device)
        encoded_coord = coords
        encoded_coord.requires_grad = False

        batch_size = 3 * 4 * 512 * 512
        out = np.empty((0, dim))

        for coord_batch in tqdm(encoded_coord.split(batch_size)):
            q = self(coord_batch).detach().cpu().numpy()
            out = np.concatenate([out, q], axis=0)
        out = out.reshape((resolution, resolution, dim))
        return out
