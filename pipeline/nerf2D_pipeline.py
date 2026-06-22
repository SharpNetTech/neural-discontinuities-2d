from pathlib import Path
import pickle
import time

import torch
import matplotlib.pyplot as plt
from PIL import Image
from largesteps.parameterize import to_differential

from geometry.mesh_triangle import TriangleMesh
from learning.nerf2d_data import prepare_data, shuffle_data
from learning.sampler import stratify_2d_offset
from neural.nerf2D_tri import MLPHybrid
from tools.plot_utils import plot_mesh, visualize_discontinuous_features, plot_slope, to_pil_image
try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)


to_jitter = True


def fit_nerf2D(image: Image, mesh: TriangleMesh, nerf2D_config: dict, fit_type: str, snapshot: Path):
    batch_size = nerf2D_config['batch_size']
    save_mask = nerf2D_config['save_mask']
    debug = nerf2D_config.get('debug', False)

    # Initialize dataset
    spp = 1

    data = prepare_data(image, spp=spp)
    total_size = data[0].shape[0]

    if debug:
        print(f'total_size: {total_size}')

    # Fix mesh
    mesh.eval()
    for param in mesh.parameters():
        param.requires_grad = False

    # Initialize MLP
    # Note the input dimension will be corrected given the positional encoding
    fea_dim = nerf2D_config['fea']
    feature_type = fit_type
    # feature_type = 'discontinuity'
    # feature_type = 'per_vertex'
    mid_dim = nerf2D_config.get('mlp_mid_dim', [128, 128])
    out_dim = nerf2D_config.get('out_dim', 3)
    mlp_hybrid = MLPHybrid(
        image, [fea_dim] + mid_dim + [out_dim], mesh,
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

    # Pre-load the data from the dataloader to save time (disables epoch shuffling)
    x = data[0].cuda()
    y = data[1].cuda()
    x_shuffled, y_shuffled = shuffle_data(x, y)

    if to_jitter:
        offset = stratify_2d_offset(x_shuffled, bin_size=torch.tensor(
            [[1.0 / spp, 1.0 / spp]]))
        offset[:, 0] = offset[:, 0] / image.height
        offset[:, 1] = offset[:, 1] / image.width
        x_shuffled = x_shuffled + offset

    x_batches = x_shuffled.split(batch_size)
    y_batches = y_shuffled.split(batch_size)

    

    # Training loop
    num_epochs = nerf2D_config['num_epochs']
    pbar = trange(num_epochs, desc='Field fitting', leave=True)
    for epoch in pbar:
        mlp_hybrid.train()  # Set the model to training mode

        to_mesh_modify = ('mesh_modify' in nerf2D_config) and (
            epoch >= nerf2D_config['mesh_modify']) and (fit_type == 'unknown_discontinuity')

        # Iterate over the training data
        start_time = time.time()
        test_timing = True
        batch_idx = 0
        for inputs, labels in zip(x_batches, y_batches):
            # Training step
            b_start_time = time.time()
            loss = mlp_hybrid.training_step(
                [inputs, labels], None, to_mesh_modify=to_mesh_modify)
            b_end_time = time.time()
            if test_timing:
                b_execution_time = b_end_time - b_start_time
                if debug:
                    print(f'Time: {b_execution_time:.4f} s')
                test_timing = False

            if to_mesh_modify and 'lambda_boundary' in mlp_hybrid.opt_config:
                loss += mlp_hybrid.opt_config['lambda_boundary'] * \
                    mlp_hybrid.boundary_loss()

            # Backpropagation and optimization
            loss.backward()
            

            # Write back to mesh
            if 'mesh_modify' in mlp_hybrid.opt_config and to_mesh_modify and \
                    (fit_type != 'per_vertex') and (fit_type != 'per_edge'):
                # Update v based on u
                mlp_hybrid.update_v()

                mesh.v = mlp_hybrid.mesh_v

            

            batch_idx += 1

        # Visualize intermediate resutls for vertex position optimization
        if 'mesh_modify' in mlp_hybrid.opt_config and mlp_hybrid.opt_config['mesh_modify']:
            if epoch >= nerf2D_config['mesh_modify'] and to_mesh_modify:
                mlp_hybrid.eval()
                with torch.no_grad():
                    x_ = torch.vstack(x_batches)
                    # soft_pixel_ras = mlp_hybrid.evaluate_raw_soft(x_=x_)
                    soft_pixel_ras = mlp_hybrid.evaluate_raw(x_=x_)
                    soft_pixel_ras = soft_pixel_ras.reshape(
                        -1, soft_pixel_ras.shape[-1])
                    l2_vis = ((torch.vstack(y_batches) -
                               soft_pixel_ras)**2).sum(dim=-1)
                    
                    l2_image = torch.zeros(
                        [image.width, image.height], dtype=torch.float32, device=mlp_hybrid.device)

                    x_ = x_ * \
                        torch.tensor([image.height, image.width],
                                     device=mlp_hybrid.device)
                    samples_pixel = torch.floor(x_).int()
                    samples_pixel[:, 0] = torch.clamp(
                        samples_pixel[:, 0], 0, image.height - 1)
                    samples_pixel[:, 1] = torch.clamp(
                        samples_pixel[:, 1], 0, image.width - 1)
                    l2_image[samples_pixel[:, 0], samples_pixel[:, 1]] = l2_vis

                    ax = plot_mesh(mesh, discontinuity=False)
                    img = ax.imshow(l2_image.cpu().numpy(), extent=[
                        0, l2_image.shape[0], l2_image.shape[1], 0], cmap='plasma')
                    
                    grad = mlp_hybrid.mesh_v.grad.detach()
                    magnitude = torch.sqrt(grad[:, 0]**2 + grad[:, 0]**2)

                    # Normalize vectors to have the longest be unit length
                    grad = -grad / magnitude.max()
                    ax.quiver(mlp_hybrid.mesh_v[:, 0].cpu().numpy(), mlp_hybrid.mesh_v[:, 1].cpu().numpy(),
                              grad[:, 0].cpu().numpy(), grad[:,
                                                             1].cpu().numpy(),
                              color='r',
                              width=0.001,
                              scale=5, scale_units='inches')

                    if debug:
                        plt.colorbar(img, ax=ax)
                        plt.savefig(
                            snapshot / f'l2_{epoch}.svg', dpi=300)
                        plt.close()

                    # Visualize samples with high l2 loss as subpixel colors
                    l2_high = 0.2
                    l2_high_mask = l2_vis > l2_high
                    l2_high_samples = x_[l2_high_mask].cpu().numpy()

                    ax = plot_mesh(mesh, discontinuity=False)
                    ax.imshow(image, extent=[0, image.width, image.height, 0])
                    l2_high_colors = torch.clamp(
                        soft_pixel_ras[l2_high_mask].detach(), 0, 1).cpu().numpy()
                    ax.scatter(l2_high_samples[:, 1], l2_high_samples[:, 0],
                               c=l2_high_colors, s=0.2, linewidths=0)
                    if debug:
                        plt.savefig(
                            snapshot / f'l2_color_{epoch}.svg', dpi=300)
                        plt.close()

                    

        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

        

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
                if debug:
                    pickle_file_path = snapshot / 'model_before.pkl'
                    with open(pickle_file_path, "wb") as pickle_file:
                        pickle.dump(mlp_hybrid, pickle_file)
                mlp_hybrid.round_w(nerf2D_config['round_threshold'])

        # Redo shuffling and jittering
        x_shuffled, y_shuffled = shuffle_data(x, y)
        if to_jitter:
            offset = stratify_2d_offset(x_shuffled, bin_size=torch.tensor(
                [[1.0 / spp, 1.0 / spp]]))
            offset[:, 0] = offset[:, 0] / image.height
            offset[:, 1] = offset[:, 1] / image.width
            x_shuffled = x_shuffled + offset
        x_batches = x_shuffled.split(batch_size)
        y_batches = y_shuffled.split(batch_size)

        end_time = time.time()
        execution_time = end_time - start_time
        # Update progress bar with latest loss
        try:
            pbar.set_postfix({
                'loss': f"{float(loss):.4f}",
                'time(s)': f"{execution_time:.2f}"
            }, refresh=False)
        except Exception:
            pass
        if debug or ((epoch + 1) == num_epochs):
            print(f'Epoch {epoch} loss: {loss}; Time: {execution_time:.4f} s')
        if (debug and (epoch % 20 == 0)) or ((epoch + 1) == num_epochs):
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
                        try:
                            ax = visualize_discontinuous_features(ax, mlp_hybrid)
                        except Exception as e:
                            print("Warning: cannot visualize discontinuous features:", e)
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
    if hasattr(mlp_hybrid, 'mesh_v'):
        mlp_hybrid.mesh_v[mlp_hybrid.mesh.boundary_vid] = mlp_hybrid.v0[mlp_hybrid.mesh.boundary_vid]
        u = to_differential(mlp_hybrid.mesh.M, mlp_hybrid.mesh_v)
        mlp_hybrid.mesh.u = torch.nn.Parameter(u, requires_grad=True)
        mlp_hybrid.mesh_v = None
        mlp_hybrid.get_v()

    return mlp_hybrid
