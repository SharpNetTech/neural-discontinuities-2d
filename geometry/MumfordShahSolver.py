import argparse
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
import cv2
import sys


class MumfordShahSolver():
    def __init__(self, img, iterations=1, tol=0.1, solver_iterations=6, alpha=10, beta=0.01, gamma=1000, epsilon=0.01):
        self.tol = tol
        self.iter = iterations
        self.solver_iter = solver_iterations
        self.alpha, self.beta, self.gamma, self.epsilon = alpha, beta, gamma, epsilon
        self.g = np.float64(img) / 255.0
        self.f = self.g
        self.edges = np.zeros(img.shape)
        self.update_gradients()

    def calc_grad_x(self, img):
        return cv2.filter2D(img, cv2.CV_64F, np.array([[-1, 0, 1]]))

    def calc_grad_y(self, img):
        return cv2.filter2D(img, cv2.CV_64F, np.array([[-1, 0, 1]]).T)

    def gradients(self, img):
        return self.calc_grad_x(img), self.calc_grad_y(img)

    def update_gradients(self):
        self.grad_x, self.grad_y = self.gradients(self.f)
        self.gradient_magnitude = np.power(
            self.grad_x, 2) + np.power(self.grad_y, 2)

    def edge_linear_operator(self, input):
        v = input.reshape(*self.g.shape)

        result = np.multiply(v, self.gradient_magnitude * self.gamma + self.beta / (4 * self.epsilon)) \
            - self.epsilon * self.beta * cv2.Laplacian(v, cv2.CV_64F)
        return result.reshape(*input.shape)

    def solve_edges(self):
        size = self.g.shape[0] * self.g.shape[1]
        A = LinearOperator(
            (size, size), matvec=self.edge_linear_operator, dtype=np.float64)
        b = np.ones(size) * self.beta / (4 * self.epsilon)

        self.edges, _ = scipy.sparse.linalg.cg(
            A, b, tol=self.tol, maxiter=self.solver_iter)
        self.edges = np.power(self.edges.reshape(*self.g.shape), 2)
        return self.edges

    def image_linear_operator(self, input):
        f = input.reshape(*self.g.shape)
        x, y = self.gradients(f)

        result = self.alpha * f - self.gamma * \
            (self.calc_grad_x(np.multiply(self.edges, x)) +
             self.calc_grad_y(np.multiply(self.edges, y)))
        return result.reshape(*input.shape)

    def solve_image(self):
        size = self.g.shape[0] * self.g.shape[1]
        A = LinearOperator(
            (size, size), matvec=self.image_linear_operator, dtype=np.float64)
        b = self.alpha * self.g.reshape(size)

        self.f, _ = scipy.sparse.linalg.cg(
            A, b, tol=self.tol, maxiter=self.solver_iter)
        self.f = self.f.reshape(*self.g.shape)
        self.update_gradients()
        return self.f

    def minimize(self):
        for i in range(0, self.iter):
            edges = self.edges.copy()
            self.solve_edges()
            self.solve_image()

            if np.linalg.norm(edges - self.edges) < 0.1:
                break

        self.edges = np.power(self.edges, 0.5)
        cv2.normalize(self.edges, self.edges, 0, 255, cv2.NORM_MINMAX)
        self.edges = 255 - np.uint8(self.edges)
        self.f = (self.f * 255).astype(np.uint8)

        return self.f, self.edges


def show_image(image, name):
    img = image * 1
    cv2.imwrite(name + ".png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png', type=Path, help='input png image')
    parser.add_argument('output', type=Path, help='output png image')
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
    args = parser.parse_args()

    img = cv2.imread(str(args.png), 1)
    result, edges = [], []
    for channel in cv2.split(img):
        solver = MumfordShahSolver(channel, iterations=args.iter, tol=0.1, solver_iterations=6,
                                   alpha=args.alpha, beta=args.beta, gamma=args.gamma, epsilon=args.epsilon)
        f, v = solver.minimize()
        result.append(f)
        edges.append(v)

    f = cv2.merge(result)
    v = np.maximum(*edges)
    # print(np.mean(img.reshape(-1, 3), axis=0))

    show_image(v, str(args.output)[:-4] + "_edges")
    show_image(f, str(args.output)[:-4] + "_image")
