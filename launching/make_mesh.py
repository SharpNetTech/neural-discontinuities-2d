#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

def write_height_mesh(fn: os.PathLike, x: NDArray[np.float32], y: NDArray[np.float32], z: NDArray[np.float32]):
    assert x.shape == z.shape and y.shape == z.shape, "x, y, z must have the same shape, use meshgrid"
    height, width = z.shape

    with open(fn, "w+b") as f:
        f.write("ply\n".encode("ascii"))
        f.write("format binary_little_endian 1.0\n".encode("ascii"))
        f.write(f"element vertex {z.size}\n".encode("ascii"))
        f.write("property float x\n".encode("ascii"))
        f.write("property float y\n".encode("ascii"))
        f.write("property float z\n".encode("ascii"))
        f.write(f"element face {(height-1)*(width-1)}\n".encode("ascii"))
        f.write("property list uchar int vertex_index\n".encode("ascii"))
        f.write("end_header\n".encode("ascii"))

        # Write vertices
        for yi in range(height):
            for xi in range(width):
                f.write(np.float32(x[yi, xi]).astype('<f4').tobytes())
                f.write(np.float32(y[yi, xi]).astype('<f4').tobytes())
                f.write(np.float32(z[yi, xi]).astype('<f4').tobytes())

        # Write faces
        for yi in range(height-1):
            for xi in range(width-1):
                f.write(np.uint8(4).astype('<u1').tobytes())
                f.write(np.int32(yi*width + xi).astype('<i4').tobytes())
                f.write(np.int32(yi*width + (xi+1)).astype('<i4').tobytes())
                f.write(np.int32((yi+1)*width + (xi+1)).astype('<i4').tobytes())
                f.write(np.int32((yi+1)*width + xi).astype('<i4').tobytes())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('npy', type=Path, help='input npy')
    parser.add_argument('--xmin', type=float, default=0.0)
    parser.add_argument('--xmax', type=float, default=1.0)
    parser.add_argument('--ymin', type=float, default=0.0)
    parser.add_argument('--ymax', type=float, default=1.0)
    args = parser.parse_args()

    z = np.load(args.npy).squeeze(-1)
    h, w = z.shape

    xs = np.linspace(args.xmin, args.xmax, w)
    ys = np.linspace(args.ymin, args.ymax, h)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    write_height_mesh(args.npy.with_suffix('.ply'), xx, yy, z)