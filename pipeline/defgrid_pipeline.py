import gc
from pathlib import Path
import pickle

import torch
from largesteps.parameterize import to_differential
from largesteps.geometry import compute_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.plot_utils import plot_mesh, color_variance_image, plot_color_variance
try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)
from geometry.mesh_triangle import TriangleMesh
from geometry.continuous_remesh import prepend_dummies, calc_edges, count_non_delaunay_edge
from geometry.remeshing import remesh_full, remesh_full_combined, delaunay_triangulate
from geometry.softras import variance_loss, color_variance_px
from geometry.defgrid import (find_k_ring_vf,
                              reconstruction_loss,
                              reg_laplacian_loss, reg_v_laplacian_loss,
                              reg_area_change_loss, reg_area_loss,
                              fix_boundary_box_positions, fix_boundary_vertex_positions,
                              set_boundary_box_positions, set_boundary_vertex_positions,
                              soft_boundary_vertex_loss)


def run_defgrid(mesh: TriangleMesh, image: Image, image_norm: torch.Tensor,
                defgrid_config: dict, opt_config: dict,
                snapshot: Path):


    largesteps_reparam_config = {
        'lambda': defgrid_config['lambda_reparam'], 'cotan': defgrid_config['cotan']}
    sigma = defgrid_config['sigma']
    k_ring = defgrid_config['k_ring']
    triangulate_method = defgrid_config['triangulate_method']

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

    debug = defgrid_config.get('debug', False)
    num_epochs = opt_config['num_epochs']
    pbar = trange(num_epochs, desc='Mesh initialization - DefGrid', leave=True)
    for epoch in pbar:
        gc.collect()
        torch.cuda.empty_cache()

        # Training loop for one epoch
        # Forward pass
        optimizer.zero_grad()

        # area_loss, areas = reg_area_loss(mesh)
        if defgrid_config['lambda_area'] > 0:
            # area_loss, areas = reg_area_change_loss(mesh)
            area_loss, areas = reg_area_loss(mesh)
        else:
            area_loss = 0
            areas = []
        if not defgrid_config['area']:
            areas = []

        if triangulate_method == 'triwild_box':
            var_loss, p_tri_association, C_mean, f_color_var = variance_loss(
                mesh, image, sigma=sigma, k_ring=k_ring, f_weights=areas,
                euclidean_threshold=1,
                to_vis=False)
        else:
            if 'to_normalize' in defgrid_config:
                to_normalize = defgrid_config['to_normalize']
            var_loss, p_tri_association, C_mean, f_color_var = variance_loss(
                mesh, image, sigma=sigma, k_ring=k_ring, f_weights=areas,
                to_normalize=to_normalize,
                loss_mask=loss_mask,
                to_vis=False)
        if defgrid_config['lambda_reconst'] > 0:
            reconst_loss, f_hat = reconstruction_loss(
                image_norm, p_tri_association, C_mean, to_vis=True)
        else:
            reconst_loss = 0
        laplacian_loss = reg_v_laplacian_loss(mesh)
        boundary_loss = soft_boundary_vertex_loss(mesh)

        loss = var_loss + \
            defgrid_config['lambda_reconst'] * reconst_loss + \
            defgrid_config['lambda_area'] * area_loss + \
            defgrid_config['lambda_lap'] * laplacian_loss + \
            defgrid_config['lambda_boundary'] * boundary_loss
        

        # Update progress bar
        try:
            pbar.set_postfix({
                'loss': f"{float(loss):.4f}",
                'v': f"{float(var_loss):.4f}"
            }, refresh=False)
        except Exception:
            pass
        if debug or (epoch + 1) == opt_config['num_epochs']:
            print(f'Epoch {epoch} loss: {loss}; var_loss: {var_loss}')

        # Save the initial result
        if epoch == 0 and debug:
            with torch.no_grad():
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
                plt.savefig(png_name, dpi=600)
                plt.close()

        # Backpropagation and optimization
        loss.backward()

        if mesh.u is None:
            if triangulate_method == 'triwild_box':
                fix_boundary_box_positions(mesh)
            elif triangulate_method == 'triwild' or triangulate_method == 'triangle':
                fix_boundary_vertex_positions(mesh)

        optimizer.step()

        # Update the learning rate using the scheduler
        if scheduler:
            scheduler.step()

        # Create a mask to only consider faces with noticeable loss
        if 'loss_threshold' in defgrid_config and epoch % remesh_epoch == 0:
            loss_mask = f_color_var > defgrid_config['loss_threshold']
            loss_mask = loss_mask.to(mesh.device)
            faces = torch.nonzero(loss_mask).squeeze(1)
            f_k_ring = []
            for fid in faces.detach().cpu().numpy().tolist():
                f_k_ring_f = find_k_ring_vf(mesh.vf, mesh.f, fid, k_ring)
                f_k_ring += f_k_ring_f
            f_k_ring = list(set(f_k_ring))
            loss_mask[f_k_ring] = True

            if debug:
                print(f'Filtered faces: {mesh.f.shape[0]} -> {len(f_k_ring)}')

        if ((epoch + 1) == opt_config['num_epochs']) and (mesh.u is not None):
            # After the last update, hard set the boundary positions
            mesh.delta_v = None
            mesh.get_v()
            if triangulate_method == 'triwild_box':
                set_boundary_box_positions(mesh)
            elif triangulate_method == 'triwild' or 'triwild+' in triangulate_method or \
                    triangulate_method == 'triangle' or 'triangle+' in triangulate_method:
                set_boundary_vertex_positions(mesh)
        else:
            # This would trigger solver to update delta_v which is used for actual loss computation
            mesh.delta_v = None
            mesh.get_v()

        # Remesh
        with torch.no_grad():
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
                fig, ax = plt.subplots()
                plt.title(f'itr: {epoch} / {num_epochs}')
                png_name = snapshot / f'before_{epoch:03d}.svg'
                if (epoch + 1) == opt_config['num_epochs']:
                    png_name = snapshot / f'before_final.svg'
                ax.imshow(image, alpha=0.8, extent=[
                    0, image.width, image.height, 0])
                plot_mesh(mesh, ax=ax, discontinuity=True)
                if debug:
                    plt.savefig(png_name, dpi=600)
                plt.close()

                V = mesh.get_v()
                F = mesh.f

                if 'subdiv_loss' in defgrid_config:
                    split_loss = defgrid_config['subdiv_loss'] if epoch > defgrid_config['skip_subdiv_itr'] else -1
                else:
                    split_loss = -1
                if 'harmonic_lambda' in defgrid_config:
                    V, F = remesh_full_combined(V, F,
                                                image,
                                                collapse=True,
                                                # collapse=False,
                                                flip=True,
                                                split_loss=split_loss,
                                                max_itr=10,
                                                min_area=10,
                                                min_edge_length=5,
                                                harmonic_lambda=defgrid_config['harmonic_lambda'],
                                                vis_itr=epoch)
                else:
                    V, F = remesh_full(V, F,
                                       image,
                                       collapse=True,
                                       flip=True,
                                       split_loss=split_loss,
                                       max_itr=10,
                                       min_area=10,
                                       min_edge_length=5,
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
                    plt.savefig(png_name, dpi=600)
                    plt.close()

                

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

        if (debug and epoch % 10 == 0) or (epoch + 1) == opt_config['num_epochs']:
            with torch.no_grad():
                if not max_color_var:
                    max_color_var = 1.01 * f_color_var_cpu.max()
                ax = color_variance_image(
                    mesh, image, f_color_var_cpu, max_color_var=max_color_var)

                num_epochs = opt_config['num_epochs']
                plt.title(f'itr: {epoch} / {num_epochs}')
                png_name = snapshot / f'defgrid_{epoch:03d}.png'
                if (epoch + 1) == opt_config['num_epochs']:
                    png_name = snapshot / f'defgrid_final.png'
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
                plt.savefig(png_name, dpi=600)
                plt.close()

    pickle_file_path = snapshot / 'mesh_final.pkl'
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(mesh, pickle_file)

    return mesh
