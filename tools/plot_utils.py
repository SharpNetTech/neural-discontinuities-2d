import os

from matplotlib.path import Path as PltPath
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
import subprocess
import svgpathtools
import torch
from torch_scatter import scatter_max
import igl
from matplotlib.collections import PolyCollection

from learning.sampler import grid_sample_triangle


def to_pil_image(img):
    img = torch.clamp(img.detach().to(torch.float32), 0, 1)
    img = img.cpu().numpy()
    img = (img * 255).round().astype("uint8")
    if img.shape[-1] == 1:
        img = img[..., 0]
    return Image.fromarray(img)


def to_pil_image_latent(img):
    minValue = img.min()
    maxValue = img.max()
    latents = (img-minValue)/(maxValue-minValue)
    image = latents.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def overlay_png_svg(png_path, svg_path, out_path):
    png_image = Image.open(png_path)

    # Read the SVG file
    paths, _ = svgpathtools.svg2paths(svg_path)

    # Create a plot
    dpi = 100
    fig_width = png_image.width / dpi
    fig_height = png_image.height / dpi

    # initiate
    fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                           dpi=dpi, constrained_layout=True)
    plt.tight_layout = {'pad': 0}
    ax.axis('off')
    ax.imshow(png_image, extent=[0, png_image.width, png_image.height, 0])

    # Convert SVG paths to Matplotlib PathPatch and add them to the plot
    for vec_path in paths:
        # Convert to a format compatible with PathPatch
        codes = []
        all_points = []
        for p in vec_path:
            points = [(p.start.real, p.start.imag), (p.end.real, p.end.imag)]

            all_points.append(points[0])
            codes.append(PltPath.MOVETO)

            if isinstance(p, svgpathtools.CubicBezier):
                all_points.append((p.control1.real, p.control1.imag))
                all_points.append((p.control2.real, p.control2.imag))
                codes.append(PltPath.CURVE4)
                codes.append(PltPath.CURVE4)
            elif isinstance(p, svgpathtools.QuadraticBezier):
                all_points.append((p.control.real, p.control.imag))
                codes.append(PltPath.CURVE3)

            all_points.append(points[1])
            codes.append(PltPath.LINETO)

        all_points = np.array(all_points)
        path = PltPath(all_points, codes)
        patch = PathPatch(path, facecolor='none', edgecolor='black', lw=0.5)
        ax.add_patch(patch)

    # Set the limits and aspect ratio
    ax.set_xlim(0, png_image.width)
    ax.set_ylim(png_image.height, 0)
    ax.set_aspect('equal')

    plt.savefig(out_path, pad_inches=0, dpi=100)
    plt.close()


@torch.no_grad()
def plot_mesh(mesh, zoom=1.0, ax=None, color=[], discontinuity=False):
    # lw = 1
    lw = 0.1
    v = mesh.get_v()
    v = v.cpu()
    x, y = v[:, 0] * zoom, v[:, 1] * zoom
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        plt.tight_layout = {'pad': 0}
    if len(color) > 0:
        triang = tri.Triangulation(x, y, mesh.f)
        # ax.tripcolor(triang, color, shading='gouraud')
    ax.triplot(x, y, triangles=mesh.f.cpu(), color='black', alpha=0.5, lw=lw)
    if len(color) > 0:
        ax.scatter(x, y, c=color, s=2)

    if discontinuity:
        discont_vid = mesh.e_discont
        for i in range(len(discont_vid)):
            start_point = v[discont_vid[i, 0]]
            end_point = v[discont_vid[i, 1]]
            ax.plot([start_point[0], end_point[0]], [
                    start_point[1], end_point[1]], color='red', lw=lw)
        if len(discont_vid):
            ax.scatter(v[discont_vid[:, 0], 0],
                       v[discont_vid[:, 0], 1], color='red', s=0.1, lw=0)
            ax.scatter(v[discont_vid[:, 1], 0],
                       v[discont_vid[:, 1], 1], color='red', s=0.1, lw=0)

    ax.set_xlim([0, mesh.size[0] * zoom])
    ax.set_ylim([0, mesh.size[1] * zoom])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    return ax


@torch.no_grad()
def plot_mesh_MumfordShah(mesh, color=[], discontinuity=[], zoom=1.0, ax=None):
    lw = 0.3
    v = mesh.get_v().cpu().numpy()
    f = mesh.f.cpu().numpy()
    x, y = v[:, 0] * zoom, v[:, 1] * zoom

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        plt.tight_layout = {'pad': 0}

    if len(color) > 0:
        vertices = np.stack((v[f, 0], v[f, 1]), axis=-1)
        collection = PolyCollection(
            vertices, edgecolors=("none",), linewidths=(0,))
        collection.set_facecolor(color)
        ax.add_collection(collection)

    # if len(color) > 0:
    #     bc = igl.barycenter(v, f)
    #     ax.scatter(bc[:, 0], bc[:, 1], c=color, s=2)

    if len(discontinuity) > 0:
        e, _, _ = igl.edge_topology(v, f)
        for i in range(len(e)):
            start_point = v[e[i, 0], :]
            end_point = v[e[i, 1], :]
            ax.plot([start_point[0], end_point[0]], [
                    start_point[1], end_point[1]], color=(discontinuity[i], discontinuity[i], discontinuity[i]), lw=lw)

    ax.set_xlim([0, mesh.size[0] * zoom])
    ax.set_ylim([0, mesh.size[1] * zoom])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


@torch.no_grad()
def plot_w(mlp, zoom=1.0, ax=None):
    mesh = mlp.mesh
    v = mesh.get_v()
    v = v.cpu()
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        plt.tight_layout = {'pad': 0}

    cmap = plt.get_cmap('plasma')
    w = mlp.get_w(mlp.w.detach(), mlp.w_mask).cpu().numpy()
    norm = plt.Normalize(vmin=0, vmax=1)
    for i in range(mlp.mesh_e.shape[0]):
        color = cmap(norm(w[i]))
        vi = mlp.mesh_e[i, 0].cpu().numpy()
        vj = mlp.mesh_e[i, 1].cpu().numpy()
        start_point = v[vi]
        end_point = v[vj]
        ax.plot([start_point[0], end_point[0]], [
            start_point[1], end_point[1]], color=color, lw=2)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='sigmoid(w)', ax=ax)

    ax.set_xlim([0, mesh.size[0] * zoom])
    ax.set_ylim([0, mesh.size[1] * zoom])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    return ax


@torch.no_grad()
def plot_slope(mlp, zoom=1.0, ax=None):
    lw = 0.4

    mesh = mlp.mesh
    v = mesh.get_v()
    v = v.cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        plt.tight_layout = {'pad': 0}

    cmap = plt.get_cmap('plasma')
    ve_features_extended = torch.clamp(mlp.ve_features[:, ...], min=0)
    w = mlp.w[ve_features_extended[..., 0]]
    l_fea = mlp.features[ve_features_extended[..., 1]]
    r_fea = mlp.features[ve_features_extended[..., 2]]

    # Blend features per corner using its two adjacent corners and the corresponding alpha values
    w = mlp.get_w(w, mlp.w_mask[ve_features_extended[..., 0]])
    w_padded = torch.where(
        mlp.ve_features[..., 0].unsqueeze(-1) >= 0, w, 0)
    if 'simple_fea' in mlp.opt_config and mlp.opt_config['simple_fea']:
        slope = w_padded
    else:
        slope = w_padded * (l_fea - r_fea)
    slope_max = slope.abs().max(dim=-1).values

    eid_pack = mlp.ve_features[..., 0].ravel()
    slope_max = slope_max.ravel()
    slope_max = slope_max[eid_pack >= 0]
    eid_pack = eid_pack[eid_pack >= 0]

    e_max = torch.zeros(
        mlp.mesh_e.shape[0], dtype=torch.float32, device=w.device)
    scatter_max(src=slope_max, index=eid_pack.long(), dim=0, out=e_max)
    e_max = e_max.cpu().numpy()

    vmax = min(1, max(e_max))
    norm = plt.Normalize(vmin=0, vmax=vmax)
    lines = np.zeros((mlp.mesh_e.shape[0], 2, 2))
    colors = np.zeros((mlp.mesh_e.shape[0], 4))

    mesh_e = mlp.mesh_e.cpu().numpy()
    is_rounded = not mlp.w_mask.all()
    for i in range(mlp.mesh_e.shape[0]):
        if is_rounded and e_max[i] == 0:
            continue
        vi = mesh_e[i, 0]
        vj = mesh_e[i, 1]
        start_point = v[vi]
        end_point = v[vj]
        lines[i] = [start_point[:2], end_point[:2]]
        colors[i] = cmap(norm(e_max[i]))

    # Create a LineCollection
    line_collection = LineCollection(lines, colors=colors, linewidths=lw)
    ax.add_collection(line_collection)

    # Rounded edges
    if is_rounded:
        lines = np.zeros((mlp.mesh_e.shape[0], 2, 2))
        colors = np.zeros((mlp.mesh_e.shape[0], 4))
        for i in range(mlp.mesh_e.shape[0]):
            if is_rounded and e_max[i] != 0:
                continue
            vi = mesh_e[i, 0]
            vj = mesh_e[i, 1]
            start_point = v[vi]
            end_point = v[vj]
            lines[i] = [start_point[:2], end_point[:2]]
            colors[i] = np.array([0.1, 0.1, 0.1, 0.5])
        line_collection = LineCollection(
            lines, colors=colors, linewidths=0.5*lw)
        ax.add_collection(line_collection)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='abs slope', ax=ax)

    ax.set_xlim([0, mesh.size[0] * zoom])
    ax.set_ylim([0, mesh.size[1] * zoom])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    return ax


@torch.no_grad()
def plot_VF(v_, f_, zoom=1.0, ax=None, color=[], e=None):
    lw = 0.1

    if isinstance(v_, torch.Tensor):
        v = v_.cpu().numpy()
    else:
        v = v_
    if isinstance(f_, torch.Tensor):
        f = f_.cpu().numpy()
    else:
        f = f_

    x, y = v[:, 0] * zoom, v[:, 1] * zoom

    if ax is None:
        fig, ax = plt.subplots()

    ax.triplot(x, y, triangles=f, color='black', alpha=0.5, lw=lw)
    if len(color) > 0:
        ax.scatter(x, y, c=color, s=20)

    # print(v[f.ravel()])
    min_bbox = np.min(v[f.ravel()], axis=0)
    max_bbox = np.max(v[f.ravel()], axis=0)

    # print(min_bbox, max_bbox)
    if isinstance(e, torch.Tensor):
        for i in range(e.shape[0]):
            start_point = v[e[i, 0]]
            end_point = v[e[i, 1]]
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]], color='red', lw=lw)

    # padding = 10
    padding = 0
    ax.set_xlim([min_bbox[0] * zoom - padding, max_bbox[0] * zoom + padding])
    ax.set_ylim([min_bbox[1] * zoom - padding, max_bbox[1] * zoom + padding])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


@torch.no_grad()
def plot_mesh_3d(mesh, zoom=1.0, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    x, y, z = mesh.v[:, 0] * zoom, mesh.v[:, 1] * zoom, mesh.v[:, 2] * zoom
    ax.plot_trisurf(x, y, z,
                    triangles=mesh.f, color='blue', alpha=0.3, lw=0.2)

    ax.set_xlim([0, mesh.size[0] * zoom])
    ax.set_ylim([0, mesh.size[1] * zoom])
    ax.set_zlim([0, -len(mesh.attributes) - 1])
    ax.invert_yaxis()
    # ax.set_aspect('equal')

    return ax


@torch.no_grad()
def plot_association_func(samples, values, mesh, zoom=1.0, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    arr = np.zeros((int(mesh.size[0] * zoom + 1),
                   int(mesh.size[1] * zoom + 1)))
    arr[((samples[:, 1]) * zoom).to(dtype=torch.int).detach().cpu().numpy(),
        (samples[:, 0] * zoom).to(dtype=torch.int).detach().cpu().numpy()] = values.detach().cpu().numpy()
    img = ax.imshow(arr, cmap='plasma', alpha=0.5,
                    extent=[0, arr.shape[0], arr.shape[1], 0])
    plt.colorbar(img, ax=ax)

    ax.set_xlim([0, mesh.size[0] * zoom])
    ax.set_ylim([0, mesh.size[1] * zoom])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


@torch.no_grad()
def color_variance(mesh, lookup_mesh, ax=None):
    min_sample = 5
    f_color_var = []
    for fid in range(mesh.f.shape[0]):
        # 1. Sample points per triangle
        ff = mesh.f[fid]
        p_min, p_max = mesh.v[ff, :].min(axis=0), mesh.v[ff, :].max(axis=0)
        p_range_min = (p_max - p_min)[0:1].min()
        step_size = p_range_min / min_sample
        samples, _ = grid_sample_triangle(
            mesh.v, mesh.f, fid, c_offset=-len(mesh.attributes), step_size=step_size)

        # 2. Compute the color variance per triangle
        J, p = lookup_mesh.lookup_color(samples)
        non_zero_rows = np.any(J.toarray() != 0, axis=1)
        J = J[non_zero_rows, :]

        C = J @ p
        C_mean = C.mean(axis=0)
        C_var = (C - C_mean) ** 2
        C_var = C_var.mean(axis=0).sum()

        f_color_var.append(C_var)

    # 3. Visualize the color variance per triangle
    x, y = mesh.v[:, 0], mesh.v[:, 1]
    if ax is None:
        fig, ax = plt.subplots()

    triang = tri.Triangulation(x, y, mesh.f)
    plt.tripcolor(triang, facecolors=f_color_var, cmap='viridis',
                  edgecolors='k', shading='gouraud')

    ax.set_xlim([0, mesh.size[0]])
    ax.set_ylim([0, mesh.size[1]])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


@torch.no_grad()
def color_variance_image_fixed(mesh, image, ax=None):
    f_color_var = []
    for fid in range(mesh.f.shape[0]):
        # 1. Sample points per triangle
        samples, _ = grid_sample_triangle(
            mesh.v, mesh.f, fid, c_offset=-len(mesh.attributes), pixel_aligned=True)

        # 2. Compute the color variance per triangle
        # C = np.array([get_color_at_position(image, xy[0], xy[1])
        #               for xy in samples]) / 255.0
        image_array = np.array(image)
        if len(image_array.shape) == 2:
            image_array = image_array[..., np.newaxis]
        C = image_array[samples[:, 1].astype(
            int), samples[:, 0].astype(int)] / 255.0
        C_mean = C.mean(axis=0)
        C_var = (C - C_mean) ** 2
        C_var = C_var.mean(axis=0).sum()

        f_color_var.append(C_var)

    # 3. Visualize the color variance per triangle
    v = mesh.get_v()
    v = v.cpu()
    x, y = v[:, 0], v[:, 1]
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image, alpha=0.5, extent=[0, image.width, image.height, 0])

    triang = tri.Triangulation(x, y, mesh.f.cpu())
    plt.tripcolor(triang, facecolors=f_color_var, cmap='plasma',
                  edgecolors='k', shading='flat', alpha=0.5)
    # ax.triplot(x, y,
    #            triangles=mesh.f, color='black', alpha=0.5, lw=0.2)

    ax.set_xlim([0, mesh.size[0]])
    ax.set_ylim([0, mesh.size[1]])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


@torch.no_grad()
def color_variance_image(mesh, image, f_color_var, ax=None, max_color_var=None):
    # 3. Visualize the color variance per triangle
    v = mesh.get_v()
    v = v.cpu()
    x, y = v[:, 0], v[:, 1]
    if ax is None:
        fig, ax = plt.subplots()

    img_alpha = 0.0
    # ax.imshow(image, alpha=img_alpha)

    triang = tri.Triangulation(
        x, y, mesh.f[0:mesh.f_interior_count, :].cpu())
    if max_color_var:
        img = plt.tripcolor(triang, facecolors=f_color_var, cmap='plasma',
                            edgecolors='k', shading='flat', alpha=(1-img_alpha),
                            vmax=max_color_var, lw=0.1)
    else:
        img = plt.tripcolor(triang, facecolors=f_color_var, cmap='plasma',
                            edgecolors='k', shading='flat', alpha=(1-img_alpha),
                            lw=0.1)

    # ax.triplot(x, y,
    #            triangles=mesh.f, color='black', alpha=0.5, lw=0.2)
    plt.colorbar(img, ax=ax)

    ax.set_xlim([0, mesh.size[0]])
    ax.set_ylim([0, mesh.size[1]])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


@torch.no_grad()
def plot_color_variance(v, f, image, f_color_var, ax=None, max_color_var=None):
    # 3. Visualize the color variance per triangle
    v = v.cpu()
    x, y = v[:, 0], v[:, 1]
    if ax is None:
        fig, ax = plt.subplots()

    img_alpha = 0.0
    # ax.imshow(image, alpha=img_alpha)

    triang = tri.Triangulation(
        x, y, f.cpu())
    if max_color_var:
        img = plt.tripcolor(triang, facecolors=f_color_var, cmap='plasma',
                            edgecolors='k', shading='flat', alpha=(1-img_alpha),
                            vmax=max_color_var, lw=0.1)
    else:
        img = plt.tripcolor(triang, facecolors=f_color_var, cmap='plasma',
                            edgecolors='k', shading='flat', alpha=(1-img_alpha),
                            lw=0.1)

    # ax.triplot(x, y,
    #            triangles=mesh.f, color='black', alpha=0.5, lw=0.2)
    plt.colorbar(img, ax=ax)

    if image:
        ax.set_xlim([0, image.width])
        ax.set_ylim([0, image.height])
    else:
        ax.set_xlim([0, v[:, 0].max()])
        ax.set_ylim([0, v[:, 1].max()])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


@ torch.no_grad()
def visualize_discontinuous_features(ax, mlp):
    # Find discontinuous features
    features_CCW_all = mlp.f_features[0:mlp.mesh.f.shape[0], :]
    features_CW_all = mlp.f_features[mlp.mesh.f.shape[0]::, :]
    features_CCW = torch.where((features_CCW_all >= 0) & (
        features_CCW_all == features_CW_all), torch.tensor(-1), features_CCW_all)
    features_CW = torch.where((features_CCW_all >= 0) & (
        features_CCW_all == features_CW_all), torch.tensor(-1), features_CW_all)

    # Compute the final colors
    offset = 2

    v = mlp.mesh.get_v()
    discont_fea_colors = []
    dots = []
    for i in range(features_CCW.shape[0]):
        for j in range(features_CCW.shape[1]):
            if features_CCW[i, j] < 0:
                continue

            fea = mlp.features[features_CCW[i, j]]
            colors = mlp.mlp(fea)
            colors = torch.clamp(colors.detach(), 0, 1)
            discont_fea_colors.append(colors.cpu().numpy())

            # Find the vertex position and edge direction
            vv = v[mlp.mesh.f[i, j]]
            j_next = (j + 1) % 3
            dir = v[mlp.mesh.f[i, j_next]] - vv
            dir = dir / torch.linalg.norm(dir)
            norm = torch.tensor([-dir[1], dir[0], 0])

            dot_pos = vv.cpu().numpy() + offset * dir.cpu().numpy() + \
                offset * norm.cpu().numpy()
            dots.append(dot_pos)

    # Visualize as dots
    positions = np.vstack(dots)
    colors = np.vstack(discont_fea_colors)
    x, y = positions[:, 0], positions[:, 1]
    ax.scatter(x, y, c=colors, s=1)

    discont_fea_colors = []
    dots = []
    for i in range(features_CW.shape[0]):
        for j in range(features_CW.shape[1]):
            if features_CW[i, j] < 0:
                continue

            fea = mlp.features[features_CW[i, j]]
            colors = mlp.mlp(fea)
            colors = torch.clamp(colors.detach(), 0, 1)
            discont_fea_colors.append(colors.cpu().numpy())

            # Find the vertex position and edge direction
            vv = v[mlp.mesh.f[i, j]]
            j_next = (j + 2) % 3
            dir = v[mlp.mesh.f[i, j_next]] - vv
            dir = dir / torch.linalg.norm(dir)
            norm = torch.tensor([-dir[1], dir[0], 0])

            dot_pos = vv.cpu().numpy() + offset * dir.cpu().numpy() - \
                offset * norm.cpu().numpy()
            dots.append(dot_pos)

    # Visualize as triangles
    positions = np.vstack(dots)
    colors = np.vstack(discont_fea_colors)
    x, y = positions[:, 0], positions[:, 1]
    ax.scatter(x, y, c=colors, s=1.5, marker='^')

    return ax


def plot_discontinuous_func(x, y, thetas, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    indices = np.argsort(x, axis=0)
    x_sorted = x[indices]
    y_reordered = y[indices]

    color_cycle = ax._get_lines.prop_cycler
    next_color = next(color_cycle)['color']

    vlines = []
    for i in range(thetas.shape[0] + 1):
        upper = thetas[i] if i < thetas.shape[0] else 1.0
        lower = thetas[i - 1]
        if i > 0:
            indices = np.where(
                (x_sorted >= lower) & (x_sorted < upper))
        else:
            indices = np.where((x_sorted < upper))
        ax.plot(x_sorted[indices], y_reordered[indices], color=next_color)

        if len(vlines) > 0 and len(y_reordered[indices]) > 0:
            vlines[-1].append(y_reordered[indices][0])
        if i < thetas.shape[0] and len(y_reordered[indices]) > 0:
            vlines.append([thetas[i], y_reordered[indices][-1]])

    for vl in vlines:
        ax.vlines(x=vl[0], ymin=vl[1], ymax=vl[2],
                  color='red', linestyles='dashed')

    return ax


def tex_graphics(path: str, size_ratio=1) -> str:
    if path == '' or not os.path.exists(path):
        return '\\small(missing)'
    height = 6 * size_ratio
    width = 6 * size_ratio
    abs_path = os.path.abspath(path)

    return fr'\includegraphics[height={height}in, width={width}in, keepaspectratio]{{{abs_path}}}'
