#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray

class BelheFactory:
    _feature_vertices: NDArray[np.float32]
    _feature_edges: NDArray[np.int32]

    def __init__(self) -> None:
        super().__init__()
        self._feature_vertices = np.array([
            # For the line
            [0.27721, 0.42631],
            [0.40487, 0.79068],
            # For the polygon
            [0.45444, 0.42631],
            [0.83864, 0.39781],
            [0.91548, 0.81547],
            [0.40487, 0.57752],
        ], dtype=np.float32)
        self._feature_edges = np.array([
            [0, 1],
            [2, 3], [3, 4], [4, 5], [5, 2],
        ], dtype=np.int32)

    @staticmethod
    def _C0_2d(pt: NDArray[np.float32], face: NDArray[np.int32]) -> NDArray[np.float32]:

        def _dot(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
            return np.sum(a*b, axis=-1, keepdims=True)

        def _cross2d(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
            a0 = a[:,:,[0]]
            a1 = a[:,:,[1]]
            b0 = b[:,:,[0]]
            b1 = b[:,:,[1]]
            return a0 * b1 - a1 * b0

        v = face - np.expand_dims(pt, axis=-2) # P x F x 2 x 2
        va = v[:,:,0,:] # P x F x 2
        vb = v[:,:,1,:]
        vd = face[:,:,1,:] - face[:,:,0,:]
        ld = np.linalg.norm(vd, axis=-1, keepdims=True) # P x F x 1
        l = np.abs(_cross2d(va, vb)) / ld
        L_ = np.square(l)
        l_ = np.sqrt(L_)

        def f(t: NDArray[np.float32]) -> NDArray[np.float32]:
            return t * (np.log(np.square(t) + L_) - 2) / 2 + l_ * np.arctan2(t, l_)

        res = f(_dot(vb,vd) / ld) - f(_dot(va,vd) / ld)
        return res.squeeze(axis=-1) / np.pi # P x F

    def forward(self, x: NDArray[np.float32], keepdims: bool=False) -> NDArray[np.float32]:
        assert x.shape[-1] == 2, "Input must have 2 elements in the last dimension"
        shape = x.shape[:-1]
        x = x.reshape(-1, 2)
        rpt = np.expand_dims(x, axis=1) # P x {1} x 2

        # A lite version of BEMquery that solves the laplacian equation.
        # The data is small enough for dense non-mollifier implementation.

        redge = np.expand_dims(
            np.take(self._feature_vertices, self._feature_edges.flatten(), axis=0).reshape(5, 2, 2),
            axis=0,
        )  # (1, num_edges, 2, 2)
        out = self._C0_2d(rpt, redge).sum(axis=-1, keepdims=True)   # P, 1

        out = out.reshape(*shape, -1)
        if not keepdims:
            out = out.squeeze(-1)
        return out

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=512, help='resolution')
    parser.add_argument('--debug', action='store_true', help='enable visualizations for debugging')
    args = parser.parse_args()

    DEBUG = args.debug

    belhe = BelheFactory()
    RESOLUTION = args.resolution
    xs = np.linspace(0, 1, RESOLUTION, endpoint=False, dtype=np.float32) + (0.5 / RESOLUTION)
    ys = np.linspace(0, 1, RESOLUTION, endpoint=False, dtype=np.float32) + (0.5 / RESOLUTION)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    pts = np.stack([xx, yy], axis=-1)  # H x W x 2

    field = belhe.forward(pts, keepdims=False)  # H x W

    # DEBUG: visualise
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.imshow(field, cmap='winter', origin='lower')
        plt.colorbar()
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), "belhe.png"), dpi=300)

    np.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), "belhe.npy"), field)
