import ast
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import os

import igl

from contextlib import contextmanager
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

def _resolve_exe(env_var: str, program_name: str, fallback: str) -> Path:
    # 1) Explicit environment variable
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value)
    # 2) On PATH
    found = shutil.which(program_name)
    if found:
        return Path(found)
    # 3) Fallback to previous default (may be user-specific)
    return Path(fallback)

# External binaries (resolved in order: ENV -> PATH -> fallback)
# Only TriWild is required in this release.
triwild_path = _resolve_exe('TRIWILD_PATH', 'TriWild', '/u6/b/chenxil/projects/TriWild/build/TriWild')

# Copied from https://github.com/Shiriluz/Word-As-Image/blob/c387b072875fc4ba8f217aa1811b7943804c7d81/code/utils.py
# pytorch adaptation of https://github.com/google/mipnerf


def parse_settings(setting_str):
    settings = {}
    try:
        pairs = setting_str.split(';')
        for pair in pairs:
            key, value = pair.split('=')
            settings[key] = ast.literal_eval(value)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing settings {value}: {e}")
        exit(1)

    return settings


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
    """Continuous learning rate decay function.
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


def compose_image(image_norm, nerf_tensor):
    # Alpha blend the nerf image with the source image
    nerf_image = nerf_tensor
    output_dim = nerf_image.shape[0]
    if output_dim > 4:
        alpha_image = nerf_image[4, :, :].unsqueeze(0).expand(4, -1, -1)
        nerf_image = alpha_image * \
            nerf_image[0:4, :, :] + \
            (1 - alpha_image) * image_norm.squeeze(0)
    return nerf_image


def get_color_at_position(image, x, y):
    width, height = image.size
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1

    # Ensure the coordinates are within the image boundaries
    x1 = min(max(x1, 0), width - 1)
    x2 = min(max(x2, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    y2 = min(max(y2, 0), height - 1)

    # Calculate fractional parts
    dx = x - x1
    dy = y - y1

    # Get the colors of the four nearest pixels
    color_tl = image.getpixel((x1, y1))
    color_tr = image.getpixel((x2, y1))
    color_bl = image.getpixel((x1, y2))
    color_br = image.getpixel((x2, y2))

    # Perform bilinear interpolation
    color = (
        color_tl[0] * (1 - dx) * (1 - dy) +
        color_tr[0] * dx * (1 - dy) +
        color_bl[0] * (1 - dx) * dy +
        color_br[0] * dx * dy,

        color_tl[1] * (1 - dx) * (1 - dy) +
        color_tr[1] * dx * (1 - dy) +
        color_bl[1] * (1 - dx) * dy +
        color_br[1] * dx * dy,

        color_tl[2] * (1 - dx) * (1 - dy) +
        color_tr[2] * dx * (1 - dy) +
        color_bl[2] * (1 - dx) * dy +
        color_br[2] * dx * dy
    )

    return color


def get_svg_size(svg_file):
    try:
        tree = ET.parse(svg_file)
    except ET.ParseError:
        print("Error:\tparsing svg failed")
        return 0, 0
    root = tree.getroot()
    width = 0
    height = 0
    if "viewBox" in root.attrib:
        if ',' in root.attrib['viewBox']:
            width = float(root.attrib['viewBox'].split(',')[2])
            height = float(root.attrib['viewBox'].split(',')[3])
        else:
            width = float(root.attrib['viewBox'].split(' ')[2])
            height = float(root.attrib['viewBox'].split(' ')[3])
    elif "width" in root.attrib and "height" in root.attrib:
        width = float(root.attrib['width'].strip('px'))
        height = float(root.attrib['height'].strip('px'))
    else:
        print("Error:\tparsing svg failed")
    return width, height


def poly2obj(polys, obj_file: Path, as_face=False):
    e2poly = {}
    e_idx = 0

    # Dedup
    if as_face:
        V = []
        F = []
        v_count = 0
        for j, poly in enumerate(polys):
            for i, v in enumerate(poly):
                V.append(np.array([[v[0], v[1], 0]]))
                if i > 0 and j == 0:
                    v_prev = v_count - 1
                    F.append(np.array([[v_prev, v_count, j]]))
                v_count += 1

        V = np.vstack(V)
        F = np.vstack(F).astype(np.int32)

        epsilon = 1e-6
        sv, svi, svj, sf = igl.remove_duplicate_vertices(V, F, epsilon)
        # nv, nf, _, _ = igl.remove_unreferenced(sv, sf)
        nv = sv
        nf = sf
        with open(obj_file, 'w') as f:
            v_count = 1
            for vi in range(nv.shape[0]):
                vv = nv[vi]
                f.write(f'v {vv[0]} {vv[1]} 0\n')
            for fi in range(nf.shape[0]):
                ff = nf[fi]
                f.write(f'f {ff[0] + 1} {ff[1] + 1}\n')
                e2poly[e_idx] = ff[2]
                e_idx += 1
    else:
        with open(obj_file, 'w') as f:
            v_count = 1
            for j, poly in enumerate(polys):
                for i, v in enumerate(poly):
                    f.write(f'v {v[0]} {v[1]} 0\n')
                    if i > 0:
                        v_prev = v_count - 1
                        if not as_face:
                            f.write(f'l {v_prev} {v_count}\n')
                        else:
                            f.write(f'f {v_prev} {v_count}\n')

                        e2poly[e_idx] = j
                        e_idx += 1
                    v_count += 1

    return e2poly


def hex_to_rgb(hex_color):
    assert hex_color[0] == '#', 'Hex color must start with #'

    # Remove the "#" character if present
    hex_color = hex_color.lstrip('#')

    # Parse the hexadecimal color into R, G, and B components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Normalize the RGB values to the [0, 1] range
    r_normalized = r / 255.0
    g_normalized = g / 255.0
    b_normalized = b / 255.0

    # Create a NumPy array of the normalized RGB values
    rgb_array = np.array([r_normalized, g_normalized, b_normalized])

    return rgb_array


@contextmanager
def temporary_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        yield Path(tmpdir)
    except:
        print('\n*\n* An error occurred. The intermediate files are left in ' +
              f'\n* "{tmpdir}".\n*\n', file=sys.stderr)
        raise
    # Unlike tempfile.TemporaryDirectory, we do not want to remove the temporary
    # directory when an exception occurs because we want to be able to manually
    # inspect the directory afterwards to see what went wrong.
    shutil.rmtree(tmpdir)


def get_state_dict(d):
    return d.get('state_dict', d)


def load_mlp(pkl_file):
    with open(pkl_file, 'rb') as file:
        # Load the object from the pickle file
        mlp_hybrid = pickle.load(file)

    # Necessary?
    # state_dict = get_state_dict(torch.load(
    #     ckpt_file, map_location=torch.device('cuda')))
    # state_dict = get_state_dict(state_dict)
    # mlp_hybrid.load_state_dict(state_dict)

    mlp_hybrid = mlp_hybrid.cuda()

    return mlp_hybrid


def measure_psnr(gt_img_path, fit_img_path, gt_mask_path=None):
    if not gt_img_path or not fit_img_path:
        return 'NA'
    if not os.path.exists(gt_img_path) or not os.path.exists(fit_img_path):
        return 'NA'

    # Load the images using PIL
    gt_img = Image.open(gt_img_path).convert('RGB')
    fit_img = Image.open(fit_img_path).convert('RGB')

    if gt_img.width != fit_img.width:
        return 'NA'

    # Convert images to numpy arrays
    gt_img_array = np.array(gt_img, dtype=np.float32)
    fit_img_array = np.array(fit_img, dtype=np.float32)

    if gt_mask_path:
        mask = Image.open(gt_mask_path).convert('L')
        mask_array = np.array(mask, dtype=np.float32)
        gt_img_array[mask_array == 0] = [255, 255, 255]

        # We also mask the fitting image
        # (note that this ignores the boundary difference)
        fit_img_array[mask_array == 0] = [255, 255, 255]

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((gt_img_array - fit_img_array) ** 2)

    # Avoid division by zero
    if mse == 0:
        return "Infinity"

    # Calculate PSNR
    pixel_max = 255.0
    psnr_value = 10 * np.log10(pixel_max**2 / mse)

    return psnr_value
