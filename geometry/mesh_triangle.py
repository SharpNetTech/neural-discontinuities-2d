import argparse
from pathlib import Path
import subprocess
import os
import time

import torch
import torch.optim as optim
from PIL import Image
import meshio

import igl
from gpytoolbox import ray_mesh_intersect
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from scipy.sparse import csr_matrix, lil_matrix

from geometry.readSVG_paths import svg2poly
from geometry.meshing import canny_init_edges
from tools.geometry_utils import flood_ext, poly_edge_correspondence
from tools.param import corresp_threshold
from tools.plot_utils import plot_mesh, plot_mesh_3d
from tools.utils import (poly2obj, temporary_directory,
                         triwild_path, hex_to_rgb,
                         learning_rate_decay, get_color_at_position)


class TriangleMesh(pl.LightningModule):
    def __init__(self, png: Path, svg: Path, triangulate_method='triwild',
                 largesteps_reparam_config={},
                 inside_only=False, to_flip=None, to_color_lookup=False,
                 canny_strong: bool = False,
                 triwild_edge_r: float = 0.01,
                 verbose: bool = False) -> None:
        super(TriangleMesh, self).__init__()

        # Control console noise
        self.verbose = verbose
        # Meshing configuration
        self._canny_strong = canny_strong
        self._triwild_edge_r = float(triwild_edge_r) if triwild_edge_r is not None else 0.01

        if svg == None:
            print('Initialized empty mesh')
            return

        self.svg2tri(png, svg, triangulate_method, inside_only,
                     to_flip, layered=to_color_lookup,
                     canny_strong=self._canny_strong,
                     target_edge_length_r=self._triwild_edge_r)

        self.largesteps_reparam_config = largesteps_reparam_config
        if len(largesteps_reparam_config) > 0:
            self.M = compute_matrix(
                self.v, self.f.to(
                    torch.int64), self.largesteps_reparam_config['lambda'],
                cotan=False if 'cotan' not in self.largesteps_reparam_config else self.largesteps_reparam_config[
                    'cotan'])
            u = to_differential(self.M, self.v)
            self.u = torch.nn.Parameter(u, requires_grad=True)
            self.delta_v = torch.zeros_like(self.v, dtype=self.v.dtype)
        else:
            self.u = None
            self.delta_v = torch.nn.Parameter(
                torch.zeros_like(self.v, dtype=self.v.dtype), requires_grad=True)
        self.tt = []

        # Cache AABB
        self.cache_p_min = []
        self.cache_p_max = []

        if to_color_lookup:
            self.J = []
            self.p = []

    def build_mesh(self, svg_info, v, f, largesteps_reparam_config={}, find_e_discont=False):
        self.polys, self.attributes, self.size = svg_info

        self.v = v
        self.f = f

        self.v_interior_count = self.v.shape[0]
        self.f_interior_count = self.f.shape[0]

        # Find edge correspondence using simple distance threshold
        if find_e_discont:
            self.e_discont, self.e_corresp = poly_edge_correspondence(
                self.polys, self, threshold=corresp_threshold)
        else:
            self.e_discont = []
            self.e_corresp = []

        # Save object boundary information
        self.register_boundary()

        self.f_np = self.f.copy()
        self.v = torch.tensor(self.v, dtype=torch.float16 if self.device == torch.device('cuda') else torch.float32,
                              requires_grad=False)
        self.f = torch.tensor(self.f, dtype=torch.int32, requires_grad=False)

        if len(largesteps_reparam_config) > 0:
            self.M = compute_matrix(
                self.v, self.f.to(
                    torch.int64), self.largesteps_reparam_config['lambda'],
                cotan=False if 'cotan' not in self.largesteps_reparam_config else self.largesteps_reparam_config[
                    'cotan'])
            u = to_differential(self.M, self.v)
            self.u = torch.nn.Parameter(u, requires_grad=True)
            self.delta_v = torch.zeros_like(self.v, dtype=self.v.dtype)
        else:
            self.u = None
            self.delta_v = torch.nn.Parameter(
                torch.zeros_like(self.v, dtype=self.v.dtype), requires_grad=True)
        self.tt = []

        # Cache AABB
        self.cache_p_min = []
        self.cache_p_max = []

    def get_v(self):
        if hasattr(self, 'u') and self.u is not None and self.delta_v is None:
            if not self.M.is_coalesced():
                self.M = self.M.coalesce()
            self.delta_v = from_differential(
                self.M, self.u, 'Cholesky') - self.v

        return self.v + self.delta_v

    def set_device(self, device):
        self.to(device)
        self.v = self.v.to(device)
        self.f = self.f.to(device)

    def configure_optimizers(self, opt_config):
        self.opt_config = opt_config

        if self.u is not None:
            # return [optim.Adam(self.parameters(),
            #                    lr=self.opt_config['learning_rate'],
            #                    betas=(0.9, 0.999) if 'betas' not in self.opt_config else self.opt_config['betas'])], \
            #     None

            opt = AdamUniform([self.u], lr=self.opt_config['lr_init']
                              if 'lr_init' in self.opt_config else self.opt_config['learning_rate'],
                              betas=(
                0.9, 0.999) if 'betas' not in self.opt_config else self.opt_config['betas'],)
            return [opt], None
        elif 'learning_rate' in self.opt_config:
            return [optim.Adam(self.parameters(),
                               lr=self.opt_config['learning_rate'],
                               betas=(0.9, 0.999) if 'betas' not in self.opt_config else self.opt_config['betas'])], \
                None
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.opt_config['lr_init'],
                betas=(
                    0.9, 0.999) if 'betas' not in self.opt_config else self.opt_config['betas'],
                eps=1e-6)

            lr_init = self.opt_config['lr_init']
            lr_final = self.opt_config['lr_final']
            lr_delay_steps = self.opt_config['lr_delay_steps']
            lr_delay_mult = self.opt_config['lr_delay_mult']

            num_epochs = self.opt_config['num_epochs']

            def lr_lambda(step): return learning_rate_decay(step, lr_init, lr_final, num_epochs,
                                                            lr_delay_steps=lr_delay_steps,
                                                            lr_delay_mult=lr_delay_mult) / lr_init

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda, last_epoch=-1)

            return [optimizer], [scheduler]

    def register_boundary(self):
        f = self.f
        if isinstance(self.f, torch.Tensor):
            f = self.f.cpu().detach().numpy()
        b = igl.boundary_facets(f)
        b = np.sort(b, axis=1)
        self.boundary_fid = []
        for i, ff in enumerate(f):
            def is_row(r):
                rr = np.sort(r)
                return np.any(np.all(b == rr, axis=1))
            if is_row(ff[0:2]) or is_row(ff[1:3]) or is_row(ff[[0, 2]]):
                self.boundary_fid.append(i)

        self.boundary_vid = []
        self.boundary_edges = []
        loops = igl.all_boundary_loop(f)

        for l in loops:
            self.boundary_edges.append([])
            for i, vid in enumerate(l):
                self.boundary_vid.append(vid)
                if i > 0:
                    self.boundary_edges[-1].append((l[i-1], l[i]))
            self.boundary_edges[-1].append([self.boundary_edges[-1]
                                           [-1][-1], self.boundary_edges[-1][0][0]])

        self.boundary_vid = torch.tensor(
            (self.boundary_vid), dtype=torch.int32, requires_grad=False)

    def triwild(self, inside_only=False, to_flip=None, layered=False, find_e_discont=False,
                target_edge_length_r: float = 0.01):
        layers = [self.polys] if not layered else [[poly]
                                                   for poly in self.polys]

        start_z = 0
        with temporary_directory() as tmp_dir:
            # Check TriWild availability
            if not Path(triwild_path).exists():
                raise FileNotFoundError(
                    f"TriWild executable not found at '{triwild_path}'.\n"
                    "Set TRIWILD_PATH to your TriWild binary or add it to PATH.")
            if not os.access(triwild_path, os.X_OK):
                print(f"Warning: TriWild at '{triwild_path}' may not be executable.")
            for i, layer in enumerate(layers):
                # 1. Output to obj
                tmp_obj = tmp_dir / f"poly_{i}.obj"
                poly2obj(layer, tmp_obj)

                # 2. Run TriWild
                cmd = [triwild_path, '--input', tmp_obj,
                       '--mute-log',
                       '--target-edge-length-r',
                       str(target_edge_length_r),  # Configurable edge length ratio
                       ]
                if inside_only:
                    cmd += ['--cut-outside']
                subprocess.run(cmd)
                if self.verbose:
                    print('TriWild done.')
                tmp_msh = tmp_dir / f"poly_{i}.obj__linear.msh"
                assert tmp_msh.exists(), 'TriWild failed to generate mesh'

                # 3. Read mesh
                if start_z == 0:
                    (self.v, self.f) = igl.read_msh(str(tmp_msh))
                    v = self.v
                    f = self.f
                else:
                    (v, f) = igl.read_msh(str(tmp_msh))

                if to_flip != None:
                    v[:, to_flip] = self.size[to_flip] - \
                        v[:, to_flip]

                # Stack polygons along the z
                v[:, 2] = start_z

                if start_z != 0:
                    self.f = np.vstack((self.f, f + self.v.shape[0]))
                    self.v = np.vstack((self.v, v))

                start_z -= 1

        self.v_interior_count = self.v.shape[0]
        self.f_interior_count = self.f.shape[0]

        # Find edge correspondence using simple distance threshold
        if find_e_discont:
            self.e_discont, self.e_corresp = poly_edge_correspondence(
                self.polys, self, threshold=corresp_threshold)
        else:
            self.e_discont = []
            self.e_corresp = []

        self.register_boundary()


    def svg2tri(self, png: Path, svg: Path, triangulate_method='triwild', inside_only=False, to_flip=None, layered=False,
                canny_strong: bool = False,
                target_edge_length_r: float = 0.01):
        triangulate_methods = triangulate_method.split('+')
        triangulate_method = triangulate_methods[0]

        self.polys, self.attributes, self.size = svg2poly(svg)

        if self.size[0] == 0 or self.size[1] == 0:
            img = Image.open(png)
            self.size = (img.width, img.height)

        # Add a boundary box for the canvas
        if not layered:
            self.polys.insert(0, [(0, 0), (self.size[0], 0),
                                  (self.size[0], self.size[1]
                                   ), (0, self.size[1]),
                                  (0, 0)])
            self.attributes.insert(0, {})
            # self.polys.append([(0, 0), (self.size[0], 0),
            #                    (self.size[0], self.size[1]), (0, self.size[1]),
            #                    (0, 0)])
            # self.attributes.append({})

        if len(triangulate_methods) > 1 and triangulate_methods[1] == 'canny':
            if self.verbose:
                print('Adding Canny detected edges')
            image = Image.open(png)
            if canny_strong:
                # Stronger preset with blur and tighter thresholds
                self.polys += canny_init_edges(image, low_threshold=50, high_threshold=150, blur=True)
            else:
                self.polys += canny_init_edges(image)

        if triangulate_method == 'triwild':
            self.triwild(inside_only=inside_only if not layered else True,
                         to_flip=to_flip, layered=layered,
                         target_edge_length_r=target_edge_length_r)
        elif triangulate_method == 'triwild_known':
            self.triwild(inside_only=inside_only if not layered else True,
                         to_flip=to_flip, layered=layered, find_e_discont=True,
                         target_edge_length_r=target_edge_length_r)
        else:
            assert False, 'Invalid triangulation method'

        self.f_np = self.f.copy()
        self.v = torch.tensor(self.v, dtype=torch.float16 if self.device == torch.device('cuda') else torch.float32,
                              requires_grad=False)
        self.f = torch.tensor(self.f, dtype=torch.int32, requires_grad=False)

    def lookup_color(self, c):
        v = self.get_v()
        c[:, 2] = -len(self.attributes)
        d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
        _, ids, _ = ray_mesh_intersect(
            c, d, v.detach().numpy(), self.f.numpy())

        valid_flags = ids >= 0
        valid_ids = ids[valid_flags]

        # target_rows = np.where(valid_flags)[0].tolist()

        p = np.zeros((len(self.attributes), 3))
        for i, attr in enumerate(self.attributes):
            cc = hex_to_rgb(attr['fill'])
            p[i, :] = cc

        # Assemble J as a sparse matrix
        hit_array = np.ones((valid_ids.shape[0]))
        row_indices = np.arange(c.shape[0])
        row_indices = row_indices[valid_flags]

        fid = self.f[valid_ids]
        valid_z = np.round(
            self.v[fid[:, 0], 2].detach().numpy(), 0).astype(int)
        column_indices = np.abs(valid_z)

        J = csr_matrix((hit_array, (row_indices, column_indices)),
                       shape=(c.shape[0], len(self.attributes)))

        return J, p

    def aabb(self, pixel_aligned=False):
        v = self.get_v()
        fid = torch.arange(self.f.shape[0])
        ff = self.f[fid]
        ff = ff.flatten()
        p_min, p_max = v[ff, :].min(
            axis=0).values, v[ff, :].max(axis=0).values

        if pixel_aligned:
            p_min = torch.floor(p_min)

        return p_min, p_max

    def lookup_tri(self, c):
        c[:, 2] = -len(self.attributes)
        d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
        _, ids, _ = ray_mesh_intersect(c, d, self.v, self.f)

        valid_flags = ids >= 0
        valid_ids = ids[valid_flags]

        return valid_ids

    def construct_Jp(self, V):
        if len(self.J) == 0 or len(self.p) == 0:
            c = np.copy(V)
            self.J, self.p = self.lookup_color(c)

    def construct_mesh_Jp(self, mesh):
        if len(self.J) == 0 or len(self.p) == 0:
            c = np.copy(mesh.v)
            self.J, self.p = self.lookup_color(c)

            # Handle boundary vertices
            if hasattr(mesh, 'e_corresp'):
                assert mesh.attributes == self.attributes
                vv2poly = np.hstack((mesh.e_corresp, mesh.e_corresp))
                # vv = mesh.e_discont.ravel().astype(int)
                # vv2poly = vv2poly.ravel().astype(int)
                vv = np.array(mesh.e_discont).ravel().astype(int)
                vv2poly = np.array(vv2poly).ravel().astype(int)

                # Modify J
                J_lil = lil_matrix(self.J)
                print(J_lil.shape, vv.shape, vv2poly.shape)
                # J_lil[vv, :] = 0
                J_lil[vv, vv2poly] = 1

                # Z sorting to keep the color of the top region
                num_rows, num_columns = J_lil.shape
                for row in range(num_rows):
                    found_first_one = False
                    for col in range(num_columns-1, -1, -1):
                        if J_lil[row, col] == 1 and not found_first_one:
                            found_first_one = True
                        elif J_lil[row, col] == 1 and found_first_one:
                            J_lil[row, col] = 0

                self.J = J_lil.tocsr()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('svg', type=Path, help='input svg')
    parser.add_argument('png', type=Path, help='input png')

    args = parser.parse_args()

    mesh = TriangleMesh(
        args.png,
        args.svg,
        triangulate_method='triwild',
        inside_only=True)
    ax = plot_mesh(mesh)

    discont_vid = mesh.e_discont

    for i in range(len(discont_vid)):
        start_point = mesh.v[discont_vid[i, 0]]
        end_point = mesh.v[discont_vid[i, 1]]
        ax.plot([start_point[0], end_point[0]], [
                start_point[1], end_point[1]], color='blue', lw=0.2)
    ax.scatter(mesh.v[discont_vid[:, 0], 0],
               mesh.v[discont_vid[:, 0], 1], color='red', s=0.5)
    ax.scatter(mesh.v[discont_vid[:, 1], 0],
               mesh.v[discont_vid[:, 1], 1], color='red', s=0.5)

    plt.savefig('discont_mesh.svg')
    plt.close()

    image = Image.open(args.png)
    C = np.array([get_color_at_position(image, xy[0], xy[1]) for xy in mesh.v])
    # plot_mesh(mesh, color=C)

    print(mesh.v.shape, mesh.f.shape, C.shape)

    exported_mesh = meshio.Mesh(
        mesh.v,
        [("triangle", mesh.f)],
        point_data={"nx": np.zeros((mesh.v.shape[0],), dtype=np.float32),
                    "ny": np.zeros((mesh.v.shape[0],), dtype=np.float32),
                    "nz": np.ones((mesh.v.shape[0],), dtype=np.float32),
                    "red": C[:, 0].astype(np.uint8),
                    "green": C[:, 1].astype(np.uint8),
                    "blue": C[:, 2].astype(np.uint8),
                    "alpha": 255 * np.ones((mesh.v.shape[0],), dtype=np.uint8)
                    }
    )
    exported_mesh.write('mesh_color.ply', file_format='ply', binary=True)

    exit()
