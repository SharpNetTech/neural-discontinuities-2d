import argparse
from pathlib import Path
import sys
import math
import numpy as np
import torch
import cv2
import scipy
from scipy.sparse.linalg import LinearOperator
import igl
from PIL import Image
from gpytoolbox import ray_mesh_intersect
from matplotlib import pyplot as plt
from tqdm import tqdm

from tools.utils import load_mlp
from tools.plot_utils import plot_mesh_MumfordShah, to_pil_image
from geometry.softras_batch import color_variance_sampling
from learning.sampler import subpixel_sample
from learning.edge_sampling_render import sum_rendering


class MumfordShahMeshSolver():
    def __init__(self, img, mesh, iterations=1, tol=0.1, solver_iterations=6, alpha=10, beta=0.01, gamma=1000, epsilon=0.01):
        self.g = np.float64(img)
        self.u = self.g

        self.iter = iterations
        self.tol = tol
        self.solver_iter = solver_iterations
        self.alpha, self.beta, self.gamma, self.epsilon = alpha, beta, gamma, epsilon

        self.v = mesh.get_v().cpu().detach().numpy()
        self.f = mesh.f.cpu().detach().numpy()
        [self.ev, self.fe, self.ef] = igl.edge_topology(self.v, self.f)
        self.edge_is_boundary = ~np.all(self.ef >= 0, axis=1)
        self.edge_lengths = np.linalg.norm(
            self.v[self.ev[:, 0]] - self.v[self.ev[:, 1]], axis=1)
        self.area = igl.doublearea(self.v, self.f) / 2
        self.bc = igl.barycenter(self.v, self.f)
        self.edges = np.zeros(self.ev.shape[0])

        self.update_gradient()

    def gradient(self, img):
        # Input: #F x 1
        # Output: #E x 1
        grad = np.zeros(self.ev.shape[0])
        for i in range(grad.shape[0]):
            if self.edge_is_boundary[i]:
                grad[i] = 0
            else:
                grad[i] = img[self.ef[i, 0]] - img[self.ef[i, 1]]
        return grad

    def update_gradient(self):
        self.grad = self.gradient(self.u)
        self.grad_mag = np.multiply(np.power(self.grad, 2), self.edge_lengths)

    def laplacian(self, discont):
        # Input: #E x 1
        # Output: #E x 1
        laplacian = np.zeros(self.ev.shape[0])
        for i in range(laplacian.shape[0]):
            v0 = self.ev[i, 0]
            v1 = self.ev[i, 1]
            if self.ef[i, 0] >= 0:
                f = self.ef[i, 0]
                bc = self.bc[f, :]
                j = np.argwhere(self.fe[f, :] == i)[0][0]
                laplacian[i] -= (discont[i] - discont[self.fe[f, (j + 2) % 3]]) * \
                    np.linalg.norm(self.v[v0, :] - bc) / self.edge_lengths[i]
                laplacian[i] -= (discont[i] - discont[self.fe[f, (j + 1) % 3]]) * \
                    np.linalg.norm(self.v[v1, :] - bc) / self.edge_lengths[i]
            if self.ef[i, 1] >= 0:
                f = self.ef[i, 1]
                bc = self.bc[f, :]
                j = np.argwhere(self.fe[f, :] == i)[0][0]
                laplacian[i] -= (discont[i] - discont[self.fe[f, (j + 1) % 3]]) * \
                    np.linalg.norm(self.v[v0, :] - bc) / self.edge_lengths[i]
                laplacian[i] -= (discont[i] - discont[self.fe[f, (j + 2) % 3]]) * \
                    np.linalg.norm(self.v[v1, :] - bc) / self.edge_lengths[i]
        return laplacian

    def divergence(self, v):
        # Input: #E x 1
        # Output: #F x 1
        div = np.zeros(self.u.shape[0])
        for i in range(div.shape[0]):
            for j in range(3):
                e = self.fe[i, j]
                if ~self.edge_is_boundary[e]:
                    sign = 1 if self.ef[e, 0] == i else -1
                    div[i] -= v[e] * sign * self.edge_lengths[e] / self.area[i]
        return div

    def edge_linear_operator(self, input):
        v = input.reshape(*self.edges.shape)

        result = np.multiply(v, self.grad_mag * self.gamma + self.beta / (
            4 * self.epsilon)) - self.epsilon * self.beta * self.laplacian(v)
        return result.reshape(*input.shape)

    def solve_edges(self):
        size = self.edges.shape[0]
        A = LinearOperator(
            (size, size), matvec=self.edge_linear_operator, dtype=np.float64)
        b = np.ones(size) * self.beta / (4 * self.epsilon)

        self.edges, _ = scipy.sparse.linalg.cg(
            A, b, tol=self.tol, maxiter=self.solver_iter)
        self.edges = np.power(self.edges, 2)
        return self.edges

    def image_linear_operator(self, input):
        u = input.reshape(*self.g.shape)
        grad = self.gradient(u)

        result = self.alpha * u - self.gamma * \
            self.divergence(np.multiply(self.edges, grad))
        return result.reshape(*input.shape)

    def solve_image(self):
        size = self.g.shape[0]
        A = LinearOperator(
            (size, size), matvec=self.image_linear_operator, dtype=np.float64)
        b = self.alpha * self.g.reshape(size)

        self.u, _ = scipy.sparse.linalg.cg(
            A, b, tol=self.tol, maxiter=self.solver_iter)
        self.u = self.u.reshape(*self.g.shape)
        self.update_gradient()
        return self.u

    def minimize(self):
        for i in tqdm(range(0, self.iter)):
            edges = self.edges.copy()
            self.solve_edges()
            self.solve_image()
            if np.linalg.norm(edges - self.edges) < 0.1:
                break

        self.edges = np.power(self.edges, 0.5)
        cv2.normalize(self.edges, self.edges, 0, 1, cv2.NORM_MINMAX)
        # self.edges = np.uint8(self.edges)
        # self.f = (self.f * 255).astype(np.uint8)

        return self.u, self.edges


def show_image(image, name):
    img = image * 1
    cv2.imwrite("snapshots/" + name + ".png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png', type=Path, help='input png image')
    parser.add_argument('output', type=Path, help='output png image')
    parser.add_argument('--mesh', type=Path, help='mesh pickle file')
    parser.add_argument('--spp', type=int, default=4, help='samples per pixel')
    parser.add_argument('--iter', type=int,
                        default=10, help='number of iterations for Mumford-Shah solver')
    parser.add_argument('--alpha', type=float, default=1,
                        help='alpha in Mumford-Shah functional')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='alpha in Mumford-Shah functional')
    parser.add_argument('--gamma', type=float, default=100,
                        help='alpha in Mumford-Shah functional')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='alpha in Mumford-Shah functional')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='alpha in Mumford-Shah functional')
    args = parser.parse_args()

    # Read input image and convert to RGB
    img = Image.open(args.png)
    img = img.convert('RGB')

    # Load mesh and sample mean color per triangle
    mlp_hybrid = load_mlp(args.mesh)
    mesh = mlp_hybrid.mesh
    C_mean, f_color_var_ret = color_variance_sampling(
        mesh.get_v(), mesh.f, img)
    color = C_mean.cpu().numpy()

    # Run Mumford-Shah with Ambrosio-Tortorelli approximation
    result, edges = [], []
    for i in range(0, color.shape[1]):
        solver = MumfordShahMeshSolver(
            color[:, i], mesh, iterations=args.iter, tol=0.1, solver_iterations=6, alpha=args.alpha, beta=args.beta, gamma=args.gamma, epsilon=args.epsilon)
        f, v = solver.minimize()
        result.append(f)
        edges.append(v)

    # Merge the results from 3 channels
    f = cv2.merge(result).reshape(-1, 3)
    v = np.clip(np.minimum(*edges), 0, 1)

    # # Plot output image and discontinuity in SVG format
    # plot_mesh_MumfordShah(mesh, color=f)
    # plt.savefig("snapshots/image_out.svg")
    # plt.close()
    plot_mesh_MumfordShah(mesh, discontinuity=v)
    vis_discont = Path(str(args.output).replace('.png', '_edges.svg'))
    plt.savefig(vis_discont)
    plt.close()

    # Save discontinuity
    e, _, _ = igl.edge_topology(
        mesh.get_v().detach().cpu().numpy(), mesh.f.detach().cpu().numpy())
    ms_discontinuity = (mesh.get_v().detach().cpu().numpy(),
                        mesh.f.detach().cpu().numpy(), e, v)
    fea_file = Path(str(args.output).replace('.png', '_edges.npz'))
    np.savez(fea_file, *ms_discontinuity)

    scale = args.scale

    # Sample within pixel
    saved_width, saved_height = img.width, img.height
    width, height = int(img.width * scale), int(img.height * scale)
    sqrt_spp = int(math.sqrt(args.spp))
    samples = subpixel_sample(width, height, sqrt_spp)
    if samples.device != mlp_hybrid.device:
        samples = samples.to(mlp_hybrid.device)

    # Samples-triangle intersection
    c = np.column_stack(
        (samples.detach().cpu().numpy(), -1 * np.ones(samples.shape[0])))
    c[:, [0, 1]] = c[:, [1, 0]]
    c[:, 0] *= mesh.size[0]
    c[:, 1] *= mesh.size[1]
    c[:, 2] = -1
    d = np.tile(np.array([[0, 0, 1]]), (c.shape[0], 1))
    _, ids, _ = ray_mesh_intersect(
        c, d, mesh.get_v().detach().cpu().numpy(), mesh.f.cpu().numpy())

    valid_flags = ids >= 0
    valid_samples = samples[valid_flags]
    valid_ids = ids[valid_flags]

    # Look up the triangle colors here based on valid_ids which has a fid per sample
    sample_colors = torch.from_numpy(f[valid_ids]).to(
        device=mlp_hybrid.device, dtype=torch.float32)

    # You need to different sum_rendering since it takes a mlp (just for the image and v, f packed in it)
    int_rendering, int_spp = sum_rendering(
        mlp_hybrid, sample_colors, valid_samples, flip_axis=True, image_size=(width, height))

    int_rendering = int_rendering / torch.clamp(int_spp, min=1).unsqueeze(-1)

    # You need to reshape this int_rendering into the image and convert it to PIL image then save
    int_rendering = int_rendering.reshape(height, width, 3)
    output_img = to_pil_image(int_rendering)
    output_img.save(args.output)
