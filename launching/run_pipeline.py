import argparse
import gc
import os
from pathlib import Path
import pickle
import torch
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from largesteps.parameterize import to_differential
from largesteps.geometry import compute_matrix

from learning.nerf2d_data import normalize_image
from tools.plot_utils import plot_mesh
from geometry.mesh_triangle import TriangleMesh
from pipeline.defgrid_pipeline import run_defgrid
from pipeline.softras_pipeline import run_softras
from pipeline.nerf2D_pipeline import fit_nerf2D
from pipeline.nerf2Dspp_pipeline import fit_nerf2D_monte_carlo
from pipeline.nerf2Dspp_samples_pipeline import fit_nerf2D_monte_carlo_samples


def set_config():
    defgrid_config = {
        'area': False,
        'sigma': 1e-1,  # SoftRas
        'k_ring': 1,
        'triangulate_method': 'triwild+canny',
        'lambda_reconst': 0.0,
        'lambda_area': 0.0,
        'lambda_lap': 0.0,

        'lambda_reparam': 1.0,
        'cotan': False,

        'lambda_boundary': 1e-2,

        'remesh_epoch': 100,
        'to_normalize': False,
        'loss_threshold': 0,
        'subdiv_loss': 2.0,  # Ref
        'skip_subdiv_itr': 10,
        'small_face_ratio': 1e-5,  # Ref

        # Combined harmonic loss
        'harmonic_lambda': 0.5,  # Ref

        'batch_size': 200 * 1024,
    }

    defgrid_opt_config = {
        'learning_rate': 1.0,
        'betas': [0.9, 0.999],

        'num_epochs': 200,
    }

    nerf2D_config = {
        'fea': 5,
        'mlp_mid_dim': [128, 128],
        'batch_size': 4 * 256 * 1024,  # Ref
        'activation': 'tanh',

        'save_mask': True,

        'num_epochs': 400, # Ref

        'learning_rate': 2 * 1e-2,  # Ref
        'l1_weight': 2e-3,

        # Round field
        'round_itr': 50,
        'round_threshold': 0.1,  # Ref

        # Modify mesh
        'mesh_modify': 70,
        'spp': [2**2, 4**2],  # 512x512
        'edge_spp': 1**2,  # 512x512

        # Boundary regularization
        'lambda_boundary': 1e-2,

        'largesteps_reparam_config': {
            'lambda_reparam': 0.5,
            'cotan': False,
        },

        # Simplified features
        'simple_fea': False,
    }

    # QSlim removed in this release
    qslim_config = None

    return {
        'defgrid_config':  defgrid_config,
        'defgrid_opt_config': defgrid_opt_config,
        'nerf2D_config': nerf2D_config,
        'qslim_config': qslim_config,
    }


if __name__ == '__main__':
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Accept a single positional [png]; svg will default to repo-root unknown.svg
    parser.add_argument('png', type=Path, help='input png')
    parser.add_argument(
        'svg',
        type=Path,
        nargs='?',
        default=(Path(__file__).resolve().parent.parent / 'unknown.svg'),
        help='input svg (default: unknown.svg at repo root)'
    )
    parser.add_argument('--mesh', type=Path, help='mesh pickle file')
    parser.add_argument('--dmesh', type=Path, help='deformed mesh pickle file')
    parser.add_argument('--model', type=Path, help='model pickle file')
    parser.add_argument('--sample', type=Path, help='sample pickle file')
    parser.add_argument('--fea', type=int, default=-1,
                        help='Feature dimension')
    parser.add_argument('--fit', type=str,
                        default='unknown_discontinuity', help='fit type')
    parser.add_argument('--snapshot', type=Path, default=Path('./snapshots'),
                        help='Text prompt')
    parser.add_argument('--depth', action='store_true',
                        help='input is depth data')
    parser.add_argument('--debug', action='store_true',
                        help='enable verbose debug logging and intermediate visualizations')
    parser.add_argument('--config-json', type=Path, default=None,
                        help='Optional JSON to override default configs; keys: defgrid_config, defgrid_opt_config, nerf2D_config')

    args = parser.parse_args()
    os.makedirs(args.snapshot, exist_ok=True)

    # Set configurations
    configs = set_config()
    defgrid_config = configs['defgrid_config']
    defgrid_opt_config = configs['defgrid_opt_config']
    nerf2D_config = configs['nerf2D_config']
    qslim_config = configs['qslim_config']

    # Optional: merge external JSON overrides
    if args.config_json is not None and args.config_json.exists():
        with open(args.config_json, 'r') as f:
            external = json.load(f)
        if 'defgrid_config' in external and external['defgrid_config']:
            defgrid_config.update(external['defgrid_config'])
        if 'defgrid_opt_config' in external and external['defgrid_opt_config']:
            defgrid_opt_config.update(external['defgrid_opt_config'])
        if 'nerf2D_config' in external and external['nerf2D_config']:
            nerf2D_config.update(external['nerf2D_config'])
        # qslim_config intentionally unused/removed

    # Propagate debug flag
    defgrid_config['debug'] = args.debug
    nerf2D_config['debug'] = args.debug

    if args.fit == 'discontinuity':
        defgrid_config['triangulate_method'] = 'triwild_known'

    if args.fea > 0:
        nerf2D_config.update({'fea': args.fea})

    fea_dim = nerf2D_config['fea']
    if args.fit == 'per_vertex':
        fea_v_dim = 3 * (1 + 2 * fea_dim) + fea_dim
        print(
            f'per_vertex: Matching DoF of unknown_discontinuity: {fea_dim} -> {fea_v_dim}')
        fea_dim = fea_v_dim
        nerf2D_config.update({'fea': fea_dim})
    elif args.fit == 'per_edge':
        fea_e_dim = 1 + 2 * fea_dim
        print(
            f'per_edge: Matching DoF of unknown_discontinuity: {fea_dim} -> {fea_e_dim}')
        fea_dim = fea_e_dim
        nerf2D_config.update({'fea': fea_dim})

    config_str = 'Configurations:\n'
    config_str += f'defgrid_config: {defgrid_config}\n'
    config_str += f'defgrid_opt_config: {defgrid_opt_config}\n'
    config_str += f'nerf2D_config: {nerf2D_config}\n'
    config_str += f'qslim_config: {qslim_config}\n'
    # with open(args.snapshot / 'config.txt', 'w') as f:
    #     f.write(config_str)
    with open(args.snapshot / 'field_config.json', 'w') as f:
        json.dump(configs, f)
    if args.debug:
        print(config_str)
        print('--' * 20)

    # Handle depth data
    meshing_png = args.png
    depth_scale = 1
    depth_flag = False
    if args.png.suffix == '.npy':
        depth_flag = True
        depth_data = np.load(args.png)

        # Save normalization scale
        depth_scale = depth_data.max()
        depth_data = depth_data / depth_data.max()

        temp_png = args.snapshot / 'depth.png'
        depth_data_scaled = (depth_data * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_data_scaled)
        depth_image.save(temp_png)

        meshing_png = temp_png

    if args.depth and args.png.suffix != '.npy':
        depth_flag = True

        image = Image.open(args.png)
        image = image.convert('L')

        depth_data = np.array(image) / 255.0
        depth_scale = depth_data.max()
        depth_data = depth_data / depth_data.max()

        temp_png = args.snapshot / 'depth.png'
        depth_data_scaled = (depth_data * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_data_scaled)
        depth_image.save(temp_png)

        meshing_png = temp_png

    # Read or generate mesh
    triangulate_method = defgrid_config['triangulate_method']
    # Meshing parameters from config (config-only, no extra CLI flags)
    canny_strong = bool(defgrid_config.get('canny_strong', False))
    triwild_edge_r = float(defgrid_config.get('triwild_edge_r', 0.01))
    if not args.mesh and not args.dmesh:
        mesh = TriangleMesh(
            meshing_png, args.svg, triangulate_method=triangulate_method, inside_only=True,
            canny_strong=canny_strong, triwild_edge_r=triwild_edge_r, verbose=args.debug)
        pickle_file_path = args.snapshot / 'mesh.pkl'
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(mesh, pickle_file)
    elif args.mesh:
        with open(args.mesh, "rb") as pickle_file:
            mesh = pickle.load(pickle_file)
            mesh.v = mesh.v + mesh.delta_v
            del mesh.delta_v  # Delete the existing Parameter
            mesh.register_buffer('delta_v', torch.zeros_like(
                mesh.v, dtype=mesh.v.dtype))
            mesh.u = None
    elif args.dmesh:
        with open(args.dmesh, "rb") as pickle_file:
            mesh = pickle.load(pickle_file)
            mesh.v = mesh.v + mesh.delta_v
            del mesh.delta_v  # Delete the existing Parameter
            mesh.register_buffer('delta_v', torch.zeros_like(
                mesh.v, dtype=mesh.v.dtype))
            mesh.u = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh.set_device(device)

    # Set up DefGrid reparameterization support
    mesh.largesteps_reparam_config = {
        'lambda': defgrid_config['lambda_reparam'],
        'cotan': defgrid_config['cotan'], }
    mesh.M = compute_matrix(
        mesh.v, mesh.f.to(
            torch.int64), mesh.largesteps_reparam_config['lambda'],
        cotan=False if 'cotan' not in mesh.largesteps_reparam_config else mesh.largesteps_reparam_config[
            'cotan'])
    u = to_differential(mesh.M, mesh.v)
    mesh.u = torch.nn.Parameter(u, requires_grad=True)
    del mesh.delta_v  # Delete the existing Parameter
    mesh.register_buffer('delta_v', torch.zeros_like(
        mesh.v, dtype=mesh.v.dtype))
    mesh.v = mesh.v.type(torch.float16 if mesh.device ==
                         torch.device('cuda') else torch.float32)
    mesh.delta_v = mesh.delta_v.type(torch.float16 if mesh.device ==
                                     torch.device('cuda') else torch.float32)
    if mesh.u is not None:
        mesh.u = mesh.u.type(torch.float16 if mesh.device ==
                             torch.device('cuda') else torch.float32)
        mesh.delta_v = None
        mesh.get_v()

    # Visualize input mesh
    if args.debug:
        ax = plot_mesh(mesh, discontinuity=True)
        plt.savefig(args.snapshot / 'mesh_in_discont.svg')
        plt.close()

    # Read image
    if not depth_flag:
        image = Image.open(args.png)
        image = image.convert('RGB')
        image_norm = torch.from_numpy(normalize_image(image)).to(device)
    else:
        image = Image.open(temp_png)
        image = image.convert('L')
        image_norm = torch.from_numpy(depth_data).to(device)

    # 1. Run DefGrid optimization
    if not args.dmesh and args.fit != 'discontinuity':
        mesh_simp = mesh
        # Run DefGrid optimization
        if 'batch_size' not in defgrid_config:
            mesh_simp_def = run_defgrid(
                mesh_simp, image, image_norm, defgrid_config, defgrid_opt_config, args.snapshot)
        else:
            mesh_simp_def = run_softras(
                mesh_simp, image, image_norm, defgrid_config, defgrid_opt_config, args.snapshot)

        del mesh_simp
        gc.collect()
        torch.cuda.empty_cache()

        pickle_file_path = args.snapshot / 'mesh_final.pkl'
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(mesh_simp_def, pickle_file)
    else:
        print('Skip QSlim and DefGrid optimization')
        mesh_simp_def = mesh

    ax = plot_mesh(mesh_simp_def, discontinuity=True)
    plt.savefig(args.snapshot / 'mesh_final.svg')
    plt.close()

    # Just in case, save the deformed mesh without delta_v
    with torch.no_grad():
        mesh_simp_def.v = mesh_simp_def.v + mesh_simp_def.delta_v
        mesh_simp_def.delta_v.zero_()
        mesh_simp_def.u = None

    # 3. Run Nerf2D fitting
    if 'spp' in nerf2D_config and (args.fit != 'per_vertex') and \
            (args.fit != 'per_edge') and (args.fit != 'discontinuity'):
        if args.sample:
            mlp_hybrid = fit_nerf2D_monte_carlo_samples(
                image, mesh_simp_def, nerf2D_config, args.fit, args.snapshot,
                model_path=args.model, sample_pickle=args.sample)
        else:
            mlp_hybrid = fit_nerf2D_monte_carlo(
                image, mesh_simp_def, nerf2D_config, args.fit, args.snapshot,
                model_path=args.model)
    else:
        mlp_hybrid = fit_nerf2D(image, mesh_simp_def, nerf2D_config,
                                args.fit, args.snapshot)

    # Save output
    # torch.save(mlp_hybrid.state_dict(), args.snapshot / 'field.pth')
    # Save the model in a pickle file
    pickle_file_path = args.snapshot / 'model.pkl'
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(mlp_hybrid, pickle_file)

    pickle_file_path = args.snapshot / 'mesh_fit.pkl'
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(mlp_hybrid.mesh, pickle_file)

    # Render with anti-alising
    spp = 4**2
    image_spp = mlp_hybrid.evaluate_spp(spp=spp)
    image_spp.save(args.snapshot / Path('fit_final_spp.png'))

    image_spp = mlp_hybrid.evaluate_spp_old(spp=spp, zoom=2)
    image_spp.save(args.snapshot / Path('fit_final_spp_2x.png'))
