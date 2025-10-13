import math
import time
import copy

import torch
from tqdm import tqdm
from largesteps.parameterize import to_differential

from learning.sampler import subpixel_sample, prepare_interior_data, important_vertices
from learning.edge_sampling_render import monte_carlo_interior_render


@torch.no_grad()
def render_finite_difference(mlp, int_samples_batches):
    # Iterate over the rendering samples
    test_timing = False
    batch_idx = 0
    int_rendering = torch.zeros([mlp.image.size[0] * mlp.image.size[1], mlp.layers[-1].out_features],
                                dtype=torch.float32, device=mlp.device)
    int_spp = torch.zeros([mlp.image.size[0] * mlp.image.size[1]],
                          dtype=mlp.mesh.f.dtype, device=mlp.device)
    # Render the interior samples
    for int_samples in int_samples_batches:
        b_start_time = time.time()

        rendering_batch, int_spp_batch = monte_carlo_interior_render(
            mlp, int_samples)
        int_rendering = int_rendering + rendering_batch
        int_spp = int_spp + int_spp_batch

        b_end_time = time.time()
        if test_timing:
            b_execution_time = b_end_time - b_start_time
            print(
                f'Interior batch rendering time: {b_execution_time:.4f} s')
            test_timing = False

        batch_idx += 1

    int_rendering = int_rendering / \
        torch.clamp(int_spp, min=1).unsqueeze(-1)

    return int_rendering


@torch.no_grad()
def loss_finite_difference(mlp, x, y, int_rendering):
    samples_img = x * \
        torch.tensor([mlp.image.size[1],
                     mlp.image.size[0]], device=mlp.mesh.device)
    samples_pixel = torch.floor(samples_img).int()
    samples_pixel[:, 0] = torch.clamp(
        samples_pixel[:, 0], 0, mlp.image.size[1] - 1)
    samples_pixel[:, 1] = torch.clamp(
        samples_pixel[:, 1], 0, mlp.image.size[0] - 1)
    y_hat = int_rendering[samples_pixel[:, 0] *
                          mlp.image.size[0] + samples_pixel[:, 1]]

    loss_func = torch.nn.MSELoss()
    loss = loss_func(y_hat, y)

    return loss


@torch.no_grad()
def edge_finite_difference(mlp, x, y, batch_size, spp, epsilon=1e-5):
    image = mlp.image

    # Generate spp samples
    sqrt_spp = int(math.sqrt(spp))
    samples = subpixel_sample(image.width, image.height, sqrt_spp)
    if samples.device != mlp.device:
        samples = samples.to(mlp.device)

    int_samples_batches = prepare_interior_data(
        spp, mlp, batch_size, int_samples_=samples)

    # Filter out vertices adjacent to discontinuous edges
    # discontinuous_vid = important_vertices(mlp, threshold=0.02).tolist()

    voi = [230, 5232, 229, 233, 186]
    e = mlp.mesh_e
    if e.device != mlp.get_v().device:
        e = e.to(mlp.get_v().device)
    e = e.long()
    samples_ei = []
    for i in voi:
        samples_ei.append(torch.nonzero((e == i).any(-1)))

    samples_ei = torch.vstack(samples_ei)
    samples_ei = samples_ei.unique()
    ee = e[samples_ei]

    discontinuous_vid = ee.unique().flatten().tolist()
    discontinuous_vid = [230, 5232, 229, 233, 186]

    # Iterate on vertices for finite difference
    v_grad = torch.zeros_like(mlp.mesh_v)
    for vi in tqdm(discontinuous_vid):
        for d in range(2):
            mlp.mesh_v[vi, d] = mlp.mesh_v[vi, d] + epsilon

            if hasattr(mlp, 'largesteps_reparam_config'):
                u = to_differential(mlp.mesh.M, mlp.mesh_v)
                mlp.mesh.u = torch.nn.Parameter(u, requires_grad=True)
            mlp.update_v()

            # Render
            int_rendering = render_finite_difference(mlp, int_samples_batches)

            # Compute loss
            loss_plus = loss_finite_difference(mlp, x, y, int_rendering)
            v_grad[vi, d] = loss_plus.item()

            mlp.mesh_v[vi, d] = mlp.mesh_v[vi, d] - epsilon

        for d in range(2):
            mlp.mesh_v[vi, d] = mlp.mesh_v[vi, d] - epsilon

            if hasattr(mlp, 'largesteps_reparam_config'):
                u = to_differential(mlp.mesh.M, mlp.mesh_v)
                mlp.mesh.u = torch.nn.Parameter(u, requires_grad=True)
            mlp.update_v()

            # Render
            int_rendering = render_finite_difference(mlp, int_samples_batches)

            # Compute loss
            loss_minus = loss_finite_difference(mlp, x, y, int_rendering)
            v_grad[vi, d] = v_grad[vi, d] - loss_minus.item()

            mlp.mesh_v[vi, d] = mlp.mesh_v[vi, d] + epsilon

    if hasattr(mlp, 'largesteps_reparam_config'):
        u = to_differential(mlp.mesh.M, mlp.mesh_v)
        mlp.mesh.u = torch.nn.Parameter(u, requires_grad=True)
    mlp.update_v()
    v_grad = v_grad / (2 * epsilon)

    return v_grad
