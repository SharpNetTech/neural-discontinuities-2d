import gc
from pathlib import Path
import pickle
import time

import torch
from largesteps.parameterize import to_differential
from largesteps.geometry import compute_matrix
from gpytoolbox import ray_mesh_intersect
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.plot_utils import plot_mesh, color_variance_image, plot_color_variance
try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)
from learning.nerf2d_data import shuffle_data
from geometry.mesh_triangle import TriangleMesh
from geometry.continuous_remesh import prepend_dummies, calc_edges, count_non_delaunay_edge
from geometry.remeshing import remesh_full, remesh_full_combined, remesh_final_clean, delaunay_triangulate
from geometry.softras_batch import (
    variance_loss, color_variance, face_mean_color, prepare_face_samples, prepare_face_bary_samples, compute_sample_weights)
from geometry.softras import color_variance_px
from geometry.defgrid import (find_k_ring_vf,
                              reconstruction_loss,
                              reg_laplacian_loss, reg_v_laplacian_loss,
                              reg_area_change_loss, reg_area_loss,
                              fix_boundary_box_positions, fix_boundary_vertex_positions,
                              set_boundary_box_positions, set_boundary_vertex_positions,
                              soft_boundary_vertex_loss)


def run_softras(mesh: TriangleMesh, image: Image, image_norm: torch.Tensor,
                defgrid_config: dict, opt_config: dict,
                snapshot: Path):
    

    largesteps_reparam_config = {
        'lambda': defgrid_config['lambda_reparam'], 'cotan': defgrid_config['cotan']}
    sigma = defgrid_config['sigma']
    k_ring = defgrid_config['k_ring']
    triangulate_method = defgrid_config['triangulate_method']
    batch_size = defgrid_config['batch_size'] if 'batch_size' in defgrid_config else image.width * image.height

    # Prepare mesh
    if not hasattr(mesh, 'u') or mesh.u is None:
        mesh.largesteps_reparam_config = largesteps_reparam_config
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
        mesh.u = mesh.u.type(torch.float16 if mesh.device ==
                             torch.device('cuda') else torch.float32)

        # Update delta_v based on u
        mesh.delta_v = None
        mesh.get_v()

    # Fit using custom for loop since the lightning one is slow
    optimizer, scheduler = mesh.configure_optimizers(opt_config)
    optimizer = optimizer[0]
    if scheduler:
        scheduler = scheduler[0]

    max_color_var = None
    loss_mask = None
    remesh_epoch = defgrid_config['remesh_epoch'] if 'remesh_epoch' in defgrid_config else -1

    dump_init = True
    vis_epoch = 20
    debug = defgrid_config.get('debug', False)
    num_epochs = opt_config['num_epochs']
    pbar = trange(num_epochs, desc='Mesh initialization - SoftRas', leave=True)
    for epoch in pbar:
        # Prepare samples
        b_samples, in_faces = prepare_face_bary_samples(mesh)
        sample_weights = compute_sample_weights(mesh.get_v(), mesh.f, in_faces)

        b_samples_shuffled, in_faces_shuffled = shuffle_data(
            b_samples, in_faces)

        b_samples_batches = b_samples_shuffled.split(batch_size)
        in_faces_batches = in_faces_shuffled.split(batch_size)

        # Recompute face mean colors after update
        with torch.no_grad():
            C_mean_in = face_mean_color(
                mesh, image, b_samples, in_faces, is_barycentric=True)

        start_time = time.time()
        batch_count = 0
        for b_samples_b, in_faces_b in zip(b_samples_batches, in_faces_batches):
            gc.collect()
            torch.cuda.empty_cache()

            # Compute sample weights based on current faces (with changing areas)
            with torch.no_grad():
                sample_weights_b = compute_sample_weights(
                    mesh.get_v(), mesh.f, in_faces_b)

            # Training loop for one epoch
            # Forward pass
            optimizer.zero_grad()

            torch.cuda.synchronize()
            loss_start_time = time.time()

            with torch.no_grad():
                w_samples_b = b_samples_b[:, 0].unsqueeze(-1) * mesh.get_v()[mesh.f[in_faces_b][:, 0], :] + \
                    b_samples_b[:, 1].unsqueeze(-1) * mesh.get_v()[mesh.f[in_faces_b][:, 1], :] + \
                    b_samples_b[:, 2].unsqueeze(-1) * \
                    mesh.get_v()[mesh.f[in_faces_b][:, 2], :]
            var_loss, _, _, _ = variance_loss(
                mesh, image,
                w_samples_b, in_faces_b, sample_weights_b, C_mean_in,
                sigma=sigma, k_ring=k_ring,
                loss_mask=loss_mask,
                to_vis=False)

            boundary_loss = soft_boundary_vertex_loss(mesh)

            loss = var_loss + defgrid_config['lambda_boundary'] * boundary_loss

            torch.cuda.synchronize()
            loss_end_time = time.time()
            int_execution_time = loss_end_time - loss_start_time
            if debug:
                print(f'\tForward time: {int_execution_time:.4f} s')

            # Save the initial result
            if dump_init and debug:
                dump_init = False
                with torch.no_grad():
                    _, f_color_var = color_variance(
                        mesh.get_v(), mesh.f, b_samples, in_faces, sample_weights, image,
                        face_mask=None, is_barycentric=True, f2px=mesh.f2px)
                    f_color_var_cpu = f_color_var.cpu().numpy()
                    if not max_color_var:
                        max_color_var = 1.01 * f_color_var_cpu.max()
                    ax = color_variance_image(
                        mesh, image, f_color_var_cpu, max_color_var=max_color_var)

                    num_epochs = opt_config['num_epochs']
                    plt.title(f'itr: {epoch} / {num_epochs}')
                    png_name = snapshot / f'defgrid_init.png'
                    plt.savefig(png_name, dpi=600)
                    plt.close()

                    fig, ax = plt.subplots()
                    plt.title(f'itr: {epoch} / {num_epochs}')
                    png_name = snapshot / f'defgrid_in_init.svg'
                    ax.imshow(image, alpha=0.8, extent=[
                        0, image.width, image.height, 0])
                    plot_mesh(mesh, ax=ax, discontinuity=True)
                    plt.savefig(png_name, dpi=300)
                    plt.close()

            # Backpropagation and optimization
            torch.cuda.synchronize()
            bw_start_time = time.time()
            loss.backward()
            torch.cuda.synchronize()
            bw_end_time = time.time()
            int_execution_time = bw_end_time - bw_start_time
            if debug:
                print(f'\tBackward time: {int_execution_time:.4f} s')

            if mesh.u is None:
                if triangulate_method == 'triwild_box':
                    fix_boundary_box_positions(mesh)
                elif triangulate_method == 'triwild' or triangulate_method == 'triangle':
                    fix_boundary_vertex_positions(mesh)

            optimizer.step()

            # Update the learning rate using the scheduler
            if scheduler:
                scheduler.step()

            # This would trigger solver to update delta_v which is used for actual loss computation
            mesh.delta_v = None
            mesh.get_v()

            # Update progress bar with batch info
            try:
                pbar.set_postfix({
                    'batch': f"{batch_count+1}/{len(b_samples_batches)}",
                    'v': f"{float(var_loss):.4f}"
                }, refresh=False)
            except Exception:
                pass
            if debug:
                print(
                    f'Epoch {epoch} batch {batch_count}/{len(b_samples_batches)} var_loss: {var_loss}')
            batch_count += 1

        torch.cuda.synchronize()
        end_time = time.time()
        int_execution_time = end_time - start_time
        # Show epoch time in progress bar and optionally print
        try:
            pbar.set_postfix({
                'epoch_s': f"{int_execution_time:.2f}"
            }, refresh=False)
        except Exception:
            pass
        if debug:
            print(f'Epoch time: {int_execution_time:.4f} s')

        if ((epoch + 1) == opt_config['num_epochs']) and (mesh.u is not None):
            # After the last update, hard set the boundary positions
            mesh.delta_v = None
            mesh.get_v()
            if triangulate_method == 'triwild_box':
                set_boundary_box_positions(mesh)
            elif triangulate_method == 'triwild' or 'triwild+' in triangulate_method or \
                    triangulate_method == 'triangle' or 'triangle+' in triangulate_method:
                set_boundary_vertex_positions(mesh)

        # Remesh
        with torch.no_grad():
            _, f_color_var = color_variance(
                mesh.get_v(), mesh.f, b_samples, in_faces, sample_weights, image,
                face_mask=None, is_barycentric=True, f2px=mesh.f2px)
            f_color_var_cpu = f_color_var.cpu().numpy()
            del f_color_var
            torch.cuda.empty_cache()

        if (remesh_epoch > 0 and (epoch + 1) % remesh_epoch == 0 and
                (epoch + 1) != opt_config['num_epochs']):
            # Before remeshing, set the boundary vertex positions
            if mesh.u is not None:
                mesh.delta_v = None
                mesh.get_v()
                if triangulate_method == 'triwild_box':
                    set_boundary_box_positions(mesh)
                elif triangulate_method == 'triwild' or 'triwild+' in triangulate_method or \
                        triangulate_method == 'triangle' or 'triangle+' in triangulate_method:
                    set_boundary_vertex_positions(mesh)

            if debug:
                print('Remeshing...')
            with torch.no_grad():
                if debug:
                    fig, ax = plt.subplots()
                    plt.title(f'itr: {epoch} / {num_epochs}')
                    png_name = snapshot / f'before_{epoch:03d}.svg'
                    if (epoch + 1) == opt_config['num_epochs']:
                        png_name = snapshot / f'before_final.svg'
                    ax.imshow(image, alpha=0.8, extent=[
                        0, image.width, image.height, 0])
                    plot_mesh(mesh, ax=ax, discontinuity=True)
                    
                    plt.savefig(png_name, dpi=300)
                    plt.close()

                    pickle_file_path = snapshot / f'mesh_before_{epoch:03d}.pkl'
                    with open(pickle_file_path, "wb") as pickle_file:
                        pickle.dump(mesh, pickle_file)

                V = mesh.get_v()
                F = mesh.f

                if 'subdiv_loss' in defgrid_config:
                    split_loss = defgrid_config['subdiv_loss'] if epoch > defgrid_config['skip_subdiv_itr'] else -1
                else:
                    split_loss = -1
                min_area = 2e-5 * (image.width * image.height)
                if 'small_face_ratio' in defgrid_config:
                    min_area = defgrid_config['small_face_ratio'] * \
                        (image.width * image.height)
                if debug:
                    print(f'\tCollapsing < {min_area}')
                V, F = remesh_full(V, F,
                                   image,
                                   collapse=True,
                                   flip=True,
                                   split_loss=split_loss,
                                   max_itr=10,
                                   min_area=min_area,
                                   min_edge_length=5,
                                   harmonic_lambda=defgrid_config['harmonic_lambda'],
                                   )

                V = V.cpu().numpy()
                F = F.cpu().numpy()

                mesh_tmp = TriangleMesh(png=None, svg=None, inside_only=True)
                svg_info = (mesh.polys, mesh.attributes, mesh.size)
                mesh_tmp.build_mesh(svg_info, V, F)
                device = mesh.device

                # Delete the original mesh to free up memory
                del mesh
                gc.collect()
                torch.cuda.empty_cache()
                mesh_tmp.set_device(device)
                mesh = mesh_tmp

                if debug:
                    fig, ax = plt.subplots()
                    plt.title(f'itr: {epoch} / {num_epochs}')
                    png_name = snapshot / f'after_{epoch:03d}.svg'
                    if (epoch + 1) == opt_config['num_epochs']:
                        png_name = snapshot / f'after_final.svg'
                    ax.imshow(image, alpha=0.8, extent=[
                        0, image.width, image.height, 0])
                    plot_mesh(mesh, ax=ax, discontinuity=True)
                    plt.savefig(png_name, dpi=300)
                    plt.close()

                    pickle_file_path = snapshot / f'mesh_after_{epoch:03d}.pkl'
                    with open(pickle_file_path, "wb") as pickle_file:
                        pickle.dump(mesh, pickle_file)

            mesh.largesteps_reparam_config = largesteps_reparam_config
            if len(largesteps_reparam_config) > 0:
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
                mesh.u = mesh.u.type(torch.float16 if mesh.device ==
                                     torch.device('cuda') else torch.float32)
                mesh.delta_v = None
                mesh.get_v()
            else:
                mesh.u = None

            loss_mask = None

            # Register optimizer and scheduler again
            del optimizer
            if scheduler:
                del scheduler
            gc.collect()
            torch.cuda.empty_cache()
            optimizer, scheduler = mesh.configure_optimizers(opt_config)
            optimizer = optimizer[0]
            if scheduler:
                scheduler = scheduler[0]

        if (remesh_epoch > 0 and (epoch + 1) == opt_config['num_epochs']):
            # Before remeshing, set the boundary vertex positions
            if mesh.u is not None:
                mesh.delta_v = None
                mesh.get_v()
                if triangulate_method == 'triwild_box':
                    set_boundary_box_positions(mesh)
                elif triangulate_method == 'triwild' or 'triwild+' in triangulate_method or \
                        triangulate_method == 'triangle' or 'triangle+' in triangulate_method:
                    set_boundary_vertex_positions(mesh)

            mesh.largesteps_reparam_config = largesteps_reparam_config
            if len(largesteps_reparam_config) > 0:
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
                mesh.u = mesh.u.type(torch.float16 if mesh.device ==
                                     torch.device('cuda') else torch.float32)
                mesh.delta_v = None
                mesh.get_v()
            else:
                mesh.u = None

            loss_mask = None

        if (debug and (epoch % vis_epoch == 0)) or (epoch + 1) == opt_config['num_epochs']:
            with torch.no_grad():
                if (epoch + 1) != opt_config['num_epochs']:
                    if not max_color_var:
                        max_color_var = 1.01 * f_color_var_cpu.max()
                    ax = color_variance_image(
                        mesh, image, f_color_var_cpu, max_color_var=max_color_var)
                    
                    plt.title(f'itr: {epoch} / {num_epochs}')
                    # plt.show(block=False)
                    png_name = snapshot / f'defgrid_{epoch:03d}.png'
                    if (epoch + 1) == opt_config['num_epochs']:
                        png_name = snapshot / f'defgrid_final.png'
                        # reconst_image = to_pil_image(f_hat)
                        # reconst_image.save(
                        #     snapshot / f'defgrid_reconst_final.png')
                    plt.savefig(png_name, dpi=600)
                    plt.close()

                fig, ax = plt.subplots()
                plt.title(f'itr: {epoch} / {num_epochs}')
                png_name = snapshot / f'defgrid_in_{epoch:03d}.svg'
                if (epoch + 1) == opt_config['num_epochs']:
                    png_name = snapshot / f'defgrid_in_final.svg'
                ax.imshow(image, alpha=0.8, extent=[
                    0, image.width, image.height, 0])
                plot_mesh(mesh, ax=ax, discontinuity=True)
                plt.savefig(png_name, dpi=300)
                plt.close()

    if debug:
        pickle_file_path = snapshot / 'mesh_final.pkl'
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(mesh, pickle_file)

    return mesh
