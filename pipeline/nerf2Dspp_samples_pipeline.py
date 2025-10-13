import math
from pathlib import Path
import pickle
import time

import torch
import matplotlib.pyplot as plt
from PIL import Image
from largesteps.parameterize import to_differential
# from torchviz import make_dot

from neural.round import round_w_cached
from geometry.mesh_triangle import TriangleMesh
from learning.nerf2d_data import prepare_data, prepare_pickled_data, shuffle_data
from learning.sampler import subpixel_sample, prepare_interior_data, prepare_edge_data
from learning.edge_sampling_render import (
    monte_carlo_interior_render, monte_carlo_interior_render_samples, sum_rendering,
    monte_carlo_edge_render, monte_carlo_edge_render_samples)
from learning.edge_finite_difference import edge_finite_difference
from neural.nerf2D_tri import MLPHybrid
from tools.plot_utils import plot_mesh, visualize_discontinuous_features, plot_slope
from tools.utils import load_mlp
try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)


def fit_nerf2D_monte_carlo_samples(image: Image, mesh: TriangleMesh, nerf2D_config: dict, fit_type: str, snapshot: Path,
                                   model_path: Path = None, sample_pickle: Path = None):
    batch_size = nerf2D_config['batch_size']
    save_mask = nerf2D_config['save_mask']
    debug = nerf2D_config.get('debug', False)

    # Initialize dataset
    if not sample_pickle:
        data = prepare_data(image, spp=1)
        total_size = data[0].shape[0]
    else:
        data = prepare_pickled_data(sample_pickle)
        total_size = data[0].shape[0]

    # Pre-load the data from the dataloader to save time
    x = data[0].cuda()
    y = data[1].cuda()

    if debug:
        print(f'total_size: {total_size}')

    # Step per batch when the input is pickled raw samples (to avoid CUDA OOM)
    if sample_pickle:
        x_shuffled, y_shuffled = shuffle_data(x, y)

        # Split into batches
        x_batches = x_shuffled.split(batch_size)
        y_batches = y_shuffled.split(batch_size)

    # Fix mesh
    mesh.eval()
    for param in mesh.parameters():
        param.requires_grad = False
    mesh.register_boundary()

    # Initialize MLP
    # Note the input dimension will be corrected given the positional encoding
    fea_dim = nerf2D_config['fea']
    feature_type = fit_type
    # feature_type = 'discontinuity'
    # feature_type = 'per_vertex'

    if model_path:
        mlp_hybrid = load_mlp(model_path)
        if not hasattr(mlp_hybrid, 'w_mask'):
            mlp_hybrid.w_mask = torch.ones_like(mlp_hybrid.w).bool()

        if not hasattr(mlp_hybrid, 'mesh_v') and ('mesh_modify' in nerf2D_config):
            if 'largesteps_reparam_config' in nerf2D_config:
                mlp_hybrid.config_largesteps(
                    nerf2D_config['largesteps_reparam_config'])
            else:
                v0 = mlp_hybrid.mesh.get_v().detach()
                mlp_hybrid.mesh_v = torch.nn.Parameter(v0, requires_grad=True)
                mlp_hybrid.opt_config['mesh_modify'] = nerf2D_config['mesh_modify']
    else:
        mid_dim = nerf2D_config['mlp_mid_dim']
        mlp_hybrid = MLPHybrid(
            image, [fea_dim] + mid_dim + [3], mesh,
            feature_type=feature_type,
            opt_config=nerf2D_config, snapshot_dir=snapshot)
        mlp_hybrid = mlp_hybrid.cuda()

    if save_mask:
        mask = mlp_hybrid.mask()
        mask.save(snapshot / 'mask.png')

    # Get optimizer
    optimizer, scheduler = mlp_hybrid.configure_optimizers()
    optimizer = optimizer[0]
    if scheduler:
        scheduler = scheduler[0]

    # Get render data
    spp = nerf2D_config['spp']
    spp_fine = 0
    e_spp = nerf2D_config['edge_spp']

    if isinstance(spp, list):
        spp_fine = spp[1]
        spp = spp[0]
    else:
        spp_fine = spp

    sqrt_spp = int(math.sqrt(spp))
    samples = subpixel_sample(image.width, image.height, sqrt_spp)
    if samples.device != mlp_hybrid.device:
        samples = samples.to(mlp_hybrid.device)

    int_samples_batches = prepare_interior_data(
        spp, mlp_hybrid, batch_size, int_samples_=samples)
    importance_threshold = (
        -1) if 'round_threshold' not in nerf2D_config else nerf2D_config['round_threshold']
    edge_samples_batches = prepare_edge_data(
        e_spp, mlp_hybrid, batch_size, importance_threshold=-1)

    # torch.autograd.set_detect_anomaly(True)

    # Training loop
    num_epochs = nerf2D_config['num_epochs']
    vis_epoch = 20
    pbar = trange(num_epochs, desc='Field fitting (input spp, edge sampling)', leave=True)
    for epoch in pbar:
        mlp_hybrid.train()  # Set the model to training mode

        to_mesh_modify = ('mesh_modify' in nerf2D_config) and (
            epoch >= nerf2D_config['mesh_modify'])
        to_smooth = ('smooth_weight' in nerf2D_config) and (
            nerf2D_config['smooth_weight'] > 0) and (epoch < nerf2D_config['round_itr'])

        for p_batch in range(len(x_batches)):
            loss = 0

            # Iterate over the rendering samples
            # torch.cuda.synchronize()
            start_time = time.time()
            test_timing = True
            batch_idx = 0
            samples_img = x * \
                torch.tensor([mlp_hybrid.image.size[1],
                              mlp_hybrid.image.size[0]], device=mlp_hybrid.mesh.device)
            samples_pixel = torch.floor(samples_img).int()
            samples_pixel[:, 0] = torch.clamp(
                samples_pixel[:, 0], 0, mlp_hybrid.image.size[1] - 1)
            samples_pixel[:, 1] = torch.clamp(
                samples_pixel[:, 1], 0, mlp_hybrid.image.size[0] - 1)

            # Exactly match the x positions
            samples_accum, y_batch = x_batches[p_batch], y_batches[p_batch]
            if to_smooth:
                samples_accum = samples_accum.requires_grad_(True)
            samples_accum = samples_accum / torch.tensor([mlp_hybrid.image.size[1],
                                                          mlp_hybrid.image.size[0]], device=mlp_hybrid.mesh.device)
            y_hat = mlp_hybrid(samples_accum)
            loss_func = torch.nn.MSELoss()
            loss += loss_func(y_hat, y_batch)

            if (mlp_hybrid.feature_type == 'unknown_discontinuity') and ('l1_weight' in mlp_hybrid.opt_config):
                loss += mlp_hybrid.opt_config['l1_weight'] * \
                    mlp_hybrid.l1_loss()

            # TODO: Add smoothness loss
            loss_record = float(loss)

            # Render and add the edge gradient
            if to_mesh_modify:
                edge_int_samples_batches = prepare_interior_data(
                    spp_fine, mlp_hybrid, batch_size)
                # Render interior for edge gradient computation
                int_rendering_e = torch.zeros([mlp_hybrid.image.size[0] * mlp_hybrid.image.size[1], mlp_hybrid.layers[-1].out_features],
                                              dtype=torch.float32, device=mlp_hybrid.device)
                int_spp_e = torch.zeros([mlp_hybrid.image.size[0] * mlp_hybrid.image.size[1]],
                                        dtype=mlp_hybrid.mesh.f.dtype, device=mlp_hybrid.device)
                with torch.no_grad():
                    # Render the interior samples
                    batch_idx = 0
                    colors_accum = []
                    samples_accum = []
                    for int_samples in edge_int_samples_batches:
                        # rendering_batch, int_spp_batch = monte_carlo_interior_render(
                        #     mlp_hybrid, int_samples)
                        # int_rendering_e = int_rendering_e + rendering_batch
                        # int_spp_e = int_spp_e + int_spp_batch

                        int_colors, int_samples_valid = monte_carlo_interior_render_samples(
                            mlp_hybrid, int_samples)
                        colors_accum.append(int_colors)
                        samples_accum.append(int_samples_valid)

                        batch_idx += 1

                    colors_accum = torch.vstack(colors_accum)
                    samples_accum = torch.vstack(samples_accum)
                    int_rendering_e, int_spp_e = sum_rendering(
                        mlp_hybrid, colors_accum, samples_accum, flip_axis=True)
                    int_rendering_e = int_rendering_e / \
                        torch.clamp(int_spp_e, min=1).unsqueeze(-1)

                edge_start_time = time.time()
                test_timing = True
                batch_idx = 0
                edge_rendering = torch.zeros([mlp_hybrid.image.size[0] * mlp_hybrid.image.size[1], mlp_hybrid.layers[-1].out_features],
                                             dtype=torch.float32, device=mlp_hybrid.device)
                edge_spp = torch.zeros([mlp_hybrid.image.size[0] * mlp_hybrid.image.size[1]],
                                       dtype=mlp_hybrid.mesh.f.dtype, device=mlp_hybrid.device)
                # Save samples then sum
                colors_accum = []
                samples_accum = []
                for edge_samples in edge_samples_batches:
                    b_start_time = time.time()

                    edge_colors, edge_samples_valid = monte_carlo_edge_render_samples(
                        mlp_hybrid, edge_samples)
                    colors_accum.append(edge_colors)
                    samples_accum.append(edge_samples_valid)

                    b_end_time = time.time()
                    if test_timing:
                        b_execution_time = b_end_time - b_start_time
                        if debug:
                            print(
                                f'Edge batch rendering time: {b_execution_time:.4f} s')
                        test_timing = False

                    batch_idx += 1

                colors_accum = torch.vstack(colors_accum)
                samples_accum = torch.vstack(samples_accum)
                edge_rendering, edge_spp = sum_rendering(
                    mlp_hybrid, colors_accum, samples_accum, flip_axis=False)

                # torch.cuda.synchronize()
                end_time = time.time()
                edge_execution_time = end_time - edge_start_time
                if debug:
                    print(
                        f'Edge rendering time: {edge_execution_time:.4f} s')

                edge_rendering = edge_rendering / \
                    torch.clamp(edge_spp, min=1).unsqueeze(-1)

                samples_indices = samples_pixel[:, 0] * \
                    mlp_hybrid.image.size[0] + samples_pixel[:, 1]
                with torch.no_grad():
                    y_hat_edge = torch.gather(
                        int_rendering_e, 0, samples_indices.view(-1, 1).long().expand(-1, int_rendering_e.shape[-1]))

                y_edge = torch.gather(
                    edge_rendering, 0, samples_indices.view(-1, 1).long().expand(-1, edge_rendering.shape[-1]))
                # y_edge = edge_rendering[samples_pixel[:, 0] *
                #                         mlp_hybrid.image.size[0] + samples_pixel[:, 1]]
                loss += (2 * (y_hat_edge - y) * y_edge).sum(dim=-1).mean()

                # make_dot(y_edge, params=dict(list(mlp_hybrid.named_parameters()))).render(
                #     "edge_rendering", format="png")
                # exit()
            if to_mesh_modify and 'lambda_boundary' in mlp_hybrid.opt_config:
                loss += mlp_hybrid.opt_config['lambda_boundary'] * \
                    mlp_hybrid.boundary_loss()

            # Backpropagate
            # torch.cuda.synchronize()
            bp_start_time = time.time()
            loss.backward()
            # torch.cuda.synchronize()
            bp_end_time = time.time()
            bp_execution_time = bp_end_time - bp_start_time
            if debug:
                print(f'Backpropagation time: {bp_execution_time:.4f} s')

            # Step the optimizer
            bp_start_time = time.time()
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            # torch.cuda.synchronize()
            bp_end_time = time.time()
            bp_execution_time = bp_end_time - bp_start_time
            if debug:
                print(f'Optimizer time: {bp_execution_time:.4f} s')

            # Write back to mesh
            if 'mesh_modify' in mlp_hybrid.opt_config and to_mesh_modify:
                # Update v based on u
                mlp_hybrid.update_v()

                mesh.v = mlp_hybrid.mesh_v

            # torch.cuda.synchronize()
            end_time = time.time()
            execution_time = end_time - start_time
            try:
                pbar.set_postfix({
                    'batch': f"{p_batch+1}/{len(x_batches)}",
                    'loss': f"{float(loss_record):.4f}"
                }, refresh=False)
            except Exception:
                pass
            if debug:
                print(
                    f'Epoch {epoch}, batch {p_batch} / {len(x_batches)}; loss: {loss_record}; Time: {execution_time:.4f} s')

            if ('mesh_modify' in nerf2D_config) and (
                    (epoch+1) >= nerf2D_config['mesh_modify']):
                edge_samples_batches = prepare_edge_data(
                    e_spp, mlp_hybrid, batch_size, importance_threshold=importance_threshold)

        # Re-shuffle data when runs in pickled mode
        if sample_pickle:
            x_shuffled, y_shuffled = shuffle_data(x, y)

            # Split into batches
            x_batches = x_shuffled.split(batch_size)
            y_batches = y_shuffled.split(batch_size)

        # Round the almost-continuous edges and fine-tune the field
        if 'round_itr' in nerf2D_config and \
                epoch > 0 and epoch == nerf2D_config['round_itr']:
            # (epoch % nerf2D_config['round_itr']) == 0 and (epoch + 1) != num_epochs:
            if feature_type == 'unknown_discontinuity':
                mlp_hybrid.eval()
                with torch.no_grad():
                    mlp_image = mlp_hybrid.evaluate()
                    if debug:
                        fit_png = snapshot / f'before_{epoch:03d}.png'
                        mlp_image.save(fit_png)

                    ax = plot_slope(mlp_hybrid)
                    # ax = plot_w(mlp_hybrid)
                    ax.imshow(mlp_image, extent=[
                        0, mlp_image.width, mlp_image.height, 0], alpha=0.8)
                    ax.set_title(f'itr: {epoch} / {num_epochs}')
                    if debug:
                        fea_png = snapshot / f'bfea_{epoch:03d}.png'
                        ax.figure.savefig(fea_png, dpi=400)
                        plt.close()
                pickle_file_path = snapshot / 'model_before.pkl'
                if debug:
                    with open(pickle_file_path, "wb") as pickle_file:
                        pickle.dump(mlp_hybrid, pickle_file)

                pickle_file_path = snapshot / 'opt_before.pkl'
                if debug:
                    with open(pickle_file_path, "wb") as pickle_file:
                        pickle.dump((optimizer, scheduler), pickle_file)
                if 'inc_threshold' not in nerf2D_config:
                    mlp_hybrid.round_w(nerf2D_config['round_threshold'])
                else:
                    round_int_samples_batches = torch.cat(int_samples_batches)
                    mlp_hybrid = round_w_cached(mlp_hybrid, (x, y),
                                                round_int_samples_batches,
                                                threshold=nerf2D_config['round_threshold'], inc_ratio=nerf2D_config['inc_threshold'])

                # exit()

        if (debug and (epoch % vis_epoch == 0)) or (epoch + 1) == num_epochs:
            if not ((epoch + 1) != num_epochs and feature_type == 'per_vertex'):
                mlp_hybrid.eval()
                with torch.no_grad():
                    mlp_image = mlp_hybrid.evaluate()
                    fit_png = snapshot / f'fit_{epoch:03d}.png'
                    if (epoch + 1) == num_epochs:
                        fit_png = snapshot / f'fit_final.png'
                    mlp_image.save(fit_png)

                    ax = plot_mesh(mesh, discontinuity=(
                        feature_type == 'discontinuity'))
                    ax.imshow(mlp_image, extent=[
                        0, mlp_image.width, mlp_image.height, 0])
                    # ax.grid()
                    ax.set_title(f'itr: {epoch} / {num_epochs}')
                    plt_png = snapshot / f'plt_{epoch:03d}.png'
                    if (epoch + 1) == num_epochs:
                        plt_png = snapshot / f'plt_final.png'
                    ax.figure.savefig(plt_png, dpi=400)
                    plt.close()

                    # Visualize features
                    if feature_type == 'discontinuity':
                        ax = plot_mesh(mesh, discontinuity=True)
                        ax = visualize_discontinuous_features(ax, mlp_hybrid)
                        ax.set_title(f'itr: {epoch} / {num_epochs}')
                        fea_png = snapshot / f'fea_{epoch:03d}.png'
                        if (epoch + 1) == num_epochs:
                            fea_png = snapshot / f'fea_final.png'
                        ax.figure.savefig(fea_png, dpi=400)
                        plt.close()

                    if feature_type == 'unknown_discontinuity':
                        ax = plot_slope(mlp_hybrid)
                        # ax = plot_w(mlp_hybrid)
                        ax.imshow(mlp_image, extent=[
                            0, mlp_image.width, mlp_image.height, 0], alpha=0.8)
                        ax.set_title(f'itr: {epoch} / {num_epochs}')
                        fea_png = snapshot / f'fea_{epoch:03d}.png'
                        if (epoch + 1) == num_epochs:
                            fea_png = snapshot / f'fea_final.png'
                        ax.figure.savefig(fea_png, dpi=400)
                        plt.close()

    # Hard set boundary vertices to have the original positions
    mlp_hybrid.mesh_v[mlp_hybrid.mesh.boundary_vid] = mlp_hybrid.v0[mlp_hybrid.mesh.boundary_vid]
    u = to_differential(mlp_hybrid.mesh.M, mlp_hybrid.mesh_v)
    mlp_hybrid.mesh.u = torch.nn.Parameter(u, requires_grad=True)
    mlp_hybrid.mesh_v = None
    mlp_hybrid.get_v()

    return mlp_hybrid
