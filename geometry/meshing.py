import argparse
from pathlib import Path
import pickle
from copy import deepcopy
import time

import torch
import cv2
import igl.triangle
from PIL import Image
import torch
import numpy as np

from tools.utils import poly2obj


def canny_init_edges(image, low_threshold: int = 100, high_threshold: int = 200, blur: bool = False):
    image = np.array(image)
    if blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.Canny(image, low_threshold, high_threshold)
    thinned = cv2.ximgproc.thinning(image)

    def save_map(channels, png_path):
        channels_image = channels[:, :, None]
        channels_image = np.concatenate(
            [channels_image, channels_image, channels_image], axis=2)
        channels_image = Image.fromarray(channels_image)
        channels_image.save(png_path)
    # save_map(image, './canny.png')
    # save_map(thinned, './thinned.png')

    # Step 1: Put vertices at all the edge pixels
    # Find all edge pixels' coordinates
    edge_pixels = np.column_stack(np.where(thinned > 0))

    # Step 2: Connect neighboring edge pixels
    # Initialize an adjacency list to store connections between edge pixels
    adj_list = {(v[1], v[0]): [] for v in edge_pixels}
    edges = []

    seen_edges = set()

    # Function to check if a neighbor is within image bounds and is an edge pixel
    def is_valid_neighbor(x, y, max_x, max_y, thinned):
        return x >= 0 and x < max_x and y >= 0 and y < max_y and thinned[y, x] > 0

    # Directions for the 8-neighboring pixels
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                  (-1, 0), (-1, -1), (0, -1), (1, -1)]

    # Populate the adjacency list by checking neighbors for each edge pixel
    offset = 0.5
    for pixel in edge_pixels:
        y, x = pixel
        pixel = np.array([x, y])
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_neighbor(nx, ny, thinned.shape[1], thinned.shape[0], thinned):
                undir_edge = ((nx, ny), tuple(pixel)) if nx < pixel[0] or (
                    nx == pixel[0] and ny < pixel[1]) else (tuple(pixel), (nx, ny))
                if undir_edge not in seen_edges:
                    adj_list[tuple(pixel)].append((nx, ny))
                    edges.append(
                        [(pixel[0] + offset, pixel[1] + offset), (nx + offset, ny + offset)])
                    seen_edges.add(undir_edge)

    return edges


def triangulate(vertices, boundary):
    v = vertices
    if isinstance(vertices, torch.Tensor):
        v = vertices.detach().cpu().numpy()
    if v.shape[1] > 2:
        v = v[:, :2]
    v, f = igl.triangle.triangulate(v, boundary, flags='YYQ')

    nv, nf, i, j = igl.remove_unreferenced(v, f)
    assert v.shape[0] == vertices.shape[0]
    return nv, nf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('png', type=Path, help='input png')

    args = parser.parse_args()

    image = Image.open(args.png)
    edges = canny_init_edges(image)

    poly2obj(edges, args.png.parent /
             args.png.name.replace('.png', '_can.obj'))
