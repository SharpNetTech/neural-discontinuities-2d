import torch


def render_face(mlp, ids, points):
    v = mlp.get_v()

    # Differentiable computation of barycentric
    l = barycentric(points, v[mlp.mesh.f[ids]])
    cc = points

    c = points.cpu().detach().numpy()
    c[:, 2] = -1

    # Compute colors
    colors, target_rows = mlp.interpolate(ids, l, c, cc)

    return colors


def barycentric(points, v):
    """
    Check if points are inside a triangle.

    Parameters:
    - points (torch.Tensor): Coordinates of points (Nx2).
    - vertices (torch.Tensor): Coordinates of the triangle vertices (3x2).

    Returns:
    - torch.Tensor: Binary tensor indicating if each point is inside the triangle.
    """

    V0 = v[:, 0, :]
    V1 = v[:, 1, :]
    V2 = v[:, 2, :]
    detT = (V1[..., 1] - V2[..., 1]) * (V0[..., 0] - V2[..., 0]) + \
        (V2[..., 0] - V1[..., 0]) * (V0[..., 1] - V2[..., 1])

    samples_x = points[:, 0]
    samples_y = points[:, 1]
    alpha = ((V1[..., 1] - V2[..., 1]) * (samples_x - V2[..., 0]) +
             (V2[..., 0] - V1[..., 0]) * (samples_y - V2[..., 1])) / detT
    beta = ((V2[..., 1] - V0[..., 1]) * (samples_x - V2[..., 0]) +
            (V0[..., 0] - V2[..., 0]) * (samples_y - V2[..., 1])) / detT
    gamma = 1 - alpha - beta

    return torch.stack([alpha, beta, gamma], dim=-1)


def gradient(y, x, grad_outputs=None):
    # gradient of y wrt x
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def divergence(y, x):
    div = 0.0
    for i in range(y.shape[-1]):  # Iterate over the last dimension
        # Compute the gradient of y[..., i] with respect to x
        grad_i = torch.autograd.grad(
            y[..., i].sum(), x, create_graph=True, retain_graph=True)[0]
        # Sum the i-th component's gradient to the divergence
        div += grad_i[..., i]
    return div


def jacobian(y, x):
    jac = torch.stack([gradient(y[..., i], x, grad_outputs=torch.ones_like(
        y[..., i])) for i in range(y.shape[-1])], dim=-1)
    return jac


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def anisotropic_laplace(y, A, x):
    grad = gradient(y, x)
    return divergence(A * grad, x)


def hessian(y, x):
    grad = gradient(y, x)
    return jacobian(grad, x)
