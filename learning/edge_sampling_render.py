import torch
from torch_scatter import scatter_sum

from neural.nerf2D_tri import render


def monte_carlo_interior_render(mlp, int_samples):
    # 1. Render interior as usual
    # Use box filter for now
    int_colors, target_rows = render(mlp, int_samples)

    # 2. Sum up the colors
    rendering = torch.zeros([mlp.image.size[0] * mlp.image.size[1], int_colors.shape[1]],
                            dtype=torch.float32, device=mlp.mesh.device)

    with torch.no_grad():
        int_samples_img = int_samples * \
            torch.tensor([mlp.image.size[1], mlp.image.size[0]],
                         device=mlp.mesh.device)
        samples_pixel = torch.floor(int_samples_img[target_rows]).int()
        samples_pixel[:, 0] = torch.clamp(
            samples_pixel[:, 0], 0, mlp.image.size[1] - 1)
        samples_pixel[:, 1] = torch.clamp(
            samples_pixel[:, 1], 0, mlp.image.size[0] - 1)

    scatter_sum(
        src=int_colors[..., 0].ravel(),
        index=(samples_pixel[:, 0] * mlp.image.size[0] + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 0])
    scatter_sum(
        src=int_colors[..., 1].ravel(),
        index=(samples_pixel[:, 0] * mlp.image.size[0] + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 1])
    scatter_sum(
        src=int_colors[..., 2].ravel(),
        index=(samples_pixel[:, 0] * mlp.image.size[0] + samples_pixel[:, 1]).long().ravel(), out=rendering[..., 2])

    int_spp = torch.zeros([mlp.image.size[0] * mlp.image.size[1]],
                          dtype=mlp.mesh.f.dtype, device=mlp.mesh.device)
    int_counts = torch.ones_like(int_colors[..., 0]).to(mlp.mesh.f.dtype)
    scatter_sum(
        src=int_counts.ravel(),
        index=(samples_pixel[:, 0] * mlp.image.size[0] + samples_pixel[:, 1]).long().ravel(), out=int_spp)

    return rendering, int_spp


def monte_carlo_interior_render_samples(mlp, int_samples):
    # 1. Render interior as usual
    # Use box filter for now
    int_colors, target_rows = render(mlp, int_samples)

    target_rows = torch.tensor(target_rows, device=mlp.mesh.device).long()
    int_samples = torch.gather(
        int_samples, 0, target_rows.unsqueeze(-1).expand(-1, int_samples.shape[-1]))

    return int_colors, int_samples


def sum_rendering(mlp, colors_accum, samples_accum, flip_axis=False, image_size=[]):
    if len(image_size) == 0:
        image_size = [mlp.image.size[0], mlp.image.size[1]]
    x_axis = 0 if not flip_axis else 1
    y_axis = 1 if not flip_axis else 0

    # Note that samples_accum has [x, y, 0]
    with torch.no_grad():
        scale = torch.tensor([image_size[0], image_size[1]] if not flip_axis
                             else [image_size[1], image_size[0]],
                             device=mlp.device)
        if samples_accum.shape[-1] == 3:
            scale = torch.cat(
                [scale, torch.tensor([1], device=mlp.device)])
        int_samples_img = samples_accum * scale
        samples_pixel = torch.floor(int_samples_img).int()
        samples_pixel[:, x_axis] = torch.clamp(
            samples_pixel[:, x_axis], 0, image_size[0] - 1)
        samples_pixel[:, y_axis] = torch.clamp(
            samples_pixel[:, y_axis], 0, image_size[1] - 1)

    #
    rendering = torch.zeros([image_size[0] * image_size[1], colors_accum.shape[-1]],
                            dtype=torch.float32, device=mlp.device)
    colors_reshaped = colors_accum.view(-1, colors_accum.shape[-1])
    indices = (samples_pixel[:, y_axis] * image_size[0] +
               samples_pixel[:, x_axis]).long().ravel()
    rendering.scatter_add_(0, indices.unsqueeze(
        1).expand(-1, colors_reshaped.shape[1]), colors_reshaped)

    #
    with torch.no_grad():
        int_spp = torch.zeros([image_size[0] * image_size[1]],
                              dtype=torch.int32, device=mlp.device)
        int_counts = torch.ones_like(colors_accum[..., 0]).to(torch.int32)
        scatter_sum(src=int_counts.ravel(), index=indices, out=int_spp)

    return rendering, int_spp


def monte_carlo_edge_render(mlp, edge_samples_):
    (edge_samples_left, edge_samples_right, samples_ei, edge_length,
     edge_normals, edge_samples_t, edge_prob) = edge_samples_

    # 1. Render edges
    # We want the grad from vertices here
    e = mlp.mesh_e
    if e.device != mlp.device:
        e = e.to(mlp.device)

    # # Parametric edge
    # ev0 = mlp.get_v()[e[samples_ei][:, 0]]
    # ev0[:, 0] = ev0[:, 0] / mlp.image.size[0]
    # ev0[:, 1] = ev0[:, 1] / mlp.image.size[1]

    # ev1 = mlp.get_v()[e[samples_ei][:, 1]]
    # ev1[:, 0] = ev1[:, 0] / mlp.image.size[0]
    # ev1[:, 1] = ev1[:, 1] / mlp.image.size[1]
    # edge_samples = ev0 + edge_samples_t * (ev1 - ev0)

    # Implicit edge
    # Normalize to avoid float32 precision issues
    ev0 = mlp.get_v()[e[samples_ei][:, 0]]
    ev0[:, 0] = ev0[:, 0] / mlp.image.size[0]
    ev0[:, 1] = ev0[:, 1] / mlp.image.size[1]

    ev1 = mlp.get_v()[e[samples_ei][:, 1]]
    ev1[:, 0] = ev1[:, 0] / mlp.image.size[0]
    ev1[:, 1] = ev1[:, 1] / mlp.image.size[1]

    with torch.no_grad():
        edge_samples = ev0 + edge_samples_t * (ev1 - ev0)
    edge_alphas = (ev0[:, 1] - ev1[:, 1]) * edge_samples[:, 0] + (ev1[:, 0] - ev0[:, 0]
                                                                  ) * edge_samples[:, 1] + (ev0[:, 0] * ev1[:, 1] - ev1[:, 0] * ev0[:, 1])
    #

    # 2. Render the perturbed edge points
    with torch.no_grad():
        # Note that there's a pixel kernel here.
        # For now just use the simple box filter.
        # Renderer takes (y, x) normalized coordinates
        edge_colors_left, target_rows_left = render(
            mlp, edge_samples_left[:, [1, 0]])
        edge_colors_right, target_rows_right = render(
            mlp, edge_samples_right[:, [1, 0]])
        target_rows = list(
            set(target_rows_left).intersection(set(target_rows_right)))
        target_rows = sorted(target_rows)
        target_mask = torch.zeros(
            [edge_samples_left.shape[0]], dtype=torch.bool, device=mlp.mesh.device)
        target_mask[target_rows] = True
        edge_colors_left = edge_colors_left[target_mask[target_rows_left]]
        edge_colors_right = edge_colors_right[target_mask[target_rows_right]]
        assert edge_colors_left.shape[0] == edge_colors_right.shape[0]

    edge_samples = edge_samples[target_rows]
    samples_ei = samples_ei[target_rows]
    edge_prob = edge_prob[target_rows]

    # 3. Compute edge integral using Monte Carlo
    # Parametric edge
    # edge_colors = -(edge_samples * edge_normals[samples_ei]).sum(
    #     dim=-1).unsqueeze(-1) * (edge_colors_left - edge_colors_right) / edge_prob

    # Implicit edge
    with torch.no_grad():
        e_length = torch.norm(ev1 - ev0, dim=-1).unsqueeze(-1)
    edge_colors = (edge_alphas.unsqueeze(-1) / e_length)[target_rows] * (
        edge_colors_left - edge_colors_right) / edge_prob

    edge_rendering = torch.zeros([mlp.image.size[0] * mlp.image.size[1], edge_colors_left.shape[1]],
                                 dtype=edge_colors.dtype, device=mlp.mesh.device)
    edge_spp = torch.zeros([mlp.image.size[0] * mlp.image.size[1]],
                           dtype=mlp.mesh.f.dtype, device=mlp.mesh.device)
    with torch.no_grad():
        edge_samples_int = edge_samples * \
            torch.tensor([mlp.image.size[0], mlp.image.size[1], 0],
                         device=mlp.mesh.device)
        edge_samples_pixel = torch.floor(edge_samples_int).int()
        edge_samples_pixel[:, 0] = edge_samples_pixel[:, 0].clamp(
            0, mlp.image.size[0] - 1)
        edge_samples_pixel[:, 1] = edge_samples_pixel[:, 1].clamp(
            0, mlp.image.size[1] - 1)

    scatter_sum(
        src=edge_colors[..., 0].ravel(),
        index=(edge_samples_pixel[:, 1] * mlp.image.size[0] + edge_samples_pixel[:, 0]).long().ravel(), out=edge_rendering[..., 0])
    scatter_sum(
        src=edge_colors[..., 1].ravel(),
        index=(edge_samples_pixel[:, 1] * mlp.image.size[0] + edge_samples_pixel[:, 0]).long().ravel(), out=edge_rendering[..., 1])
    scatter_sum(
        src=edge_colors[..., 2].ravel(),
        index=(edge_samples_pixel[:, 1] * mlp.image.size[0] + edge_samples_pixel[:, 0]).long().ravel(), out=edge_rendering[..., 2])

    edge_counts = torch.ones_like(edge_colors[..., 0]).to(mlp.mesh.f.dtype)
    scatter_sum(
        src=edge_counts.ravel(),
        index=(edge_samples_pixel[:, 1] * mlp.image.size[0] + edge_samples_pixel[:, 0]).long().ravel(), out=edge_spp)

    return edge_rendering, edge_spp


def monte_carlo_edge_render_samples(mlp, edge_samples_):
    (edge_samples_left, edge_samples_right, samples_ei, edge_length,
     edge_normals, edge_samples_t, edge_prob) = edge_samples_

    # 1. Render edges
    # We want the grad from vertices here
    e = mlp.mesh_e
    if e.device != mlp.device:
        e = e.to(mlp.device)

    # Implicit edge
    # Normalize to avoid float32 precision issues
    ev0 = mlp.get_v()[e[samples_ei][:, 0]]
    ev0[:, 0] = ev0[:, 0] / mlp.image.size[0]
    ev0[:, 1] = ev0[:, 1] / mlp.image.size[1]

    ev1 = mlp.get_v()[e[samples_ei][:, 1]]
    ev1[:, 0] = ev1[:, 0] / mlp.image.size[0]
    ev1[:, 1] = ev1[:, 1] / mlp.image.size[1]

    with torch.no_grad():
        edge_samples = ev0 + edge_samples_t * (ev1 - ev0)
    edge_alphas = (ev0[:, 1] - ev1[:, 1]) * edge_samples[:, 0] + (ev1[:, 0] - ev0[:, 0]
                                                                  ) * edge_samples[:, 1] + (ev0[:, 0] * ev1[:, 1] - ev1[:, 0] * ev0[:, 1])
    #

    # 2. Render the perturbed edge points
    with torch.no_grad():
        # Note that there's a pixel kernel here.
        # For now just use the simple box filter.
        # Renderer takes (y, x) normalized coordinates
        edge_colors_left, target_rows_left = render(
            mlp, edge_samples_left[:, [1, 0]])
        edge_colors_right, target_rows_right = render(
            mlp, edge_samples_right[:, [1, 0]])
        target_rows = list(
            set(target_rows_left).intersection(set(target_rows_right)))
        target_rows = sorted(target_rows)
        target_rows = torch.tensor(target_rows, device=mlp.mesh.device).long()
        target_mask = torch.zeros(
            [edge_samples_left.shape[0]], dtype=torch.bool, device=mlp.mesh.device)
        target_mask[target_rows] = True
        edge_colors_left = edge_colors_left[target_mask[target_rows_left]]
        edge_colors_right = edge_colors_right[target_mask[target_rows_right]]
        assert edge_colors_left.shape[0] == edge_colors_right.shape[0]

    with torch.no_grad():
        e_length = torch.norm(ev1 - ev0, dim=-1).unsqueeze(-1)

    edge_alphas = (edge_alphas.unsqueeze(-1))
    edge_prob = edge_prob * e_length

    edge_samples = torch.gather(
        edge_samples, 0, target_rows.unsqueeze(-1).expand(-1, edge_samples.shape[-1]))
    samples_ei = torch.gather(samples_ei, 0, target_rows)
    edge_prob = torch.gather(edge_prob, 0, target_rows.view(-1, 1))
    edge_alphas = torch.gather(
        edge_alphas, 0, target_rows.view(-1, 1))

    # 3. Compute edge integral using Monte Carlo
    # Parametric edge
    # edge_colors = -(edge_samples * edge_normals[samples_ei]).sum(
    #     dim=-1).unsqueeze(-1) * (edge_colors_left - edge_colors_right) / edge_prob

    # Implicit edge
    edge_colors = edge_alphas * \
        (edge_colors_left - edge_colors_right) / edge_prob

    return edge_colors, edge_samples
