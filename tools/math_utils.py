import math
import torch


def rotation_matrix(angle):
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

    return rotation_matrix


def ainb(a, b):
    """https://stackoverflow.com/questions/50666440/column-row-slicing-a-torch-sparse-tensor"""
    """gets mask for elements of a in b"""

    size = (b.size(0), a.size(0))

    if size[0] == 0:  # Prevents error in torch.Tensor.max(dim=0)
        return torch.tensor([False]*a.size(0), dtype=torch.bool)

    a = a.expand((size[0], size[1]))
    b = b.expand((size[1], size[0])).T

    mask = a.eq(b).max(dim=0).values

    return mask


def ainb_wrapper(a, b, splits=.72):
    """https://stackoverflow.com/questions/50666440/column-row-slicing-a-torch-sparse-tensor"""
    inds = int(len(a)**splits)

    tmp = [ainb(a[i*inds:(i+1)*inds], b) for i in list(range(inds))]

    return torch.cat(tmp)


def slice_torch_sparse_coo_tensor(t, slices):
    """https://stackoverflow.com/questions/50666440/column-row-slicing-a-torch-sparse-tensor"""
    """
    params:
    -------
    t: tensor to slice
    slices: slice for each dimension

    returns:
    --------
    t[slices[0], slices[1], ..., slices[n]]
    """

    t = t.coalesce()
    # assert len(slices) == len(t.size())
    for i in range(len(slices)):
        if type(slices[i]) is not torch.Tensor:
            slices[i] = torch.tensor(slices[i], dtype=torch.long)

    indices = t.indices()
    values = t.values()
    for dim, slice in enumerate(slices):
        invert = False
        if t.size(0) * 0.6 < len(slice):
            invert = True
            all_nodes = torch.arange(t.size(0))
            unique, counts = torch.cat(
                [all_nodes, slice]).unique(return_counts=True)
            slice = unique[counts == 1]
        if slice.size(0) > 400:
            mask = ainb_wrapper(indices[dim], slice)
        else:
            mask = ainb(indices[dim], slice)
        if invert:
            mask = ~mask
        indices = indices[:, mask]
        values = values[mask]

    return torch.sparse_coo_tensor(indices, values, t.size()).coalesce()
