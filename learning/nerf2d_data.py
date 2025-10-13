import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from learning.sampler import stratify_2d


def normalize_image(image):
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    image = image.point(lambda p: p * (255.0 / 65535)
                        if image.mode == 'I;16' else p)

    image_norm = np.array(image).astype(np.float32) / 255.0
    if image_norm.ndim == 2:
        image_norm = image_norm[..., np.newaxis]

    return image_norm


def dissassemble_image(width, height, image, spp=1, to_stratify=False):
    if spp == 1:
        y_coords, x_coords = torch.meshgrid(torch.arange(
            height), torch.arange(width), indexing='ij')
    else:
        y_coords, x_coords = torch.meshgrid(torch.linspace(0, spp * height - 1, spp * height),
                                            torch.linspace(0, spp * width - 1, spp * width), indexing='ij')
        y_coords = y_coords / spp
        x_coords = x_coords / spp

    y_coords = y_coords.reshape(-1, 1)
    x_coords = x_coords.reshape(-1, 1)

    if spp > 1:
        y_coords_nn = torch.floor(y_coords).long().clamp(0, height - 1)
        x_coords_nn = torch.floor(x_coords).long().clamp(0, width - 1)
    else:
        y_coords_nn = y_coords
        x_coords_nn = x_coords

    if image.ndim == 3:
        colors = image[y_coords_nn, x_coords_nn]
        colors = torch.tensor(colors).squeeze(1)
    else:
        colors = image.squeeze(0)[:, y_coords_nn, x_coords_nn]

    if to_stratify:
        samples = torch.hstack([y_coords, x_coords])
        samples = stratify_2d(samples, bin_size=torch.tensor(
            [[1.0 / spp, 1.0 / spp]]))
        y_coords = samples[:, 0]
        x_coords = samples[:, 1]

    # Offset by 0.5 x bin
    y_coords = y_coords.float() + 1.0 / (2 * spp)
    x_coords = x_coords.float() + 1.0 / (2 * spp)

    y_coords = y_coords / height
    x_coords = x_coords / width

    assert y_coords.min() >= 0 and y_coords.max(
    ) <= 1 and x_coords.min() >= 0 and x_coords.max() <= 1

    data = [(torch.tensor([y, x]), color)
            for y, x, color in zip(y_coords, x_coords, colors)]

    return data


def dissassemble_image_chunk(width, height, image, spp=1, to_stratify=False):
    if spp == 1:
        y_coords, x_coords = torch.meshgrid(torch.arange(
            height), torch.arange(width), indexing='ij')
    else:
        y_coords, x_coords = torch.meshgrid(torch.linspace(0, spp * height - 1, spp * height),
                                            torch.linspace(0, spp * width - 1, spp * width), indexing='ij')
        y_coords = y_coords / spp
        x_coords = x_coords / spp

    y_coords = y_coords.reshape(-1, 1)
    x_coords = x_coords.reshape(-1, 1)

    if spp > 1:
        y_coords_nn = torch.floor(y_coords).long().clamp(0, height - 1)
        x_coords_nn = torch.floor(x_coords).long().clamp(0, width - 1)
    else:
        y_coords_nn = y_coords
        x_coords_nn = x_coords

    if image.ndim == 3:
        colors = image[y_coords_nn, x_coords_nn]
        colors = torch.tensor(colors).squeeze(1)
    else:
        colors = image.squeeze(0)[:, y_coords_nn, x_coords_nn]

    if to_stratify:
        samples = torch.hstack([y_coords, x_coords])
        samples = stratify_2d(samples, bin_size=torch.tensor(
            [[1.0 / spp, 1.0 / spp]]))
        y_coords = samples[:, 0]
        x_coords = samples[:, 1]

    # Offset by 0.5 x bin
    y_coords = y_coords.float() + 1.0 / (2 * spp)
    x_coords = x_coords.float() + 1.0 / (2 * spp)

    y_coords = y_coords / height
    x_coords = x_coords / width

    assert y_coords.min() >= 0 and y_coords.max(
    ) <= 1 and x_coords.min() >= 0 and x_coords.max() <= 1

    data = (torch.hstack([y_coords, x_coords]), colors)

    return data


def shuffle_data(x, y):
    indices = torch.randperm(x.shape[0])
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    return x_shuffled, y_shuffled


def shuffle_data_triple(x, y, z):
    indices = torch.randperm(x.shape[0])
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    z_shuffled = z[indices]
    return x_shuffled, y_shuffled, z_shuffled


def prepare_data(image, spp=1):
    # Normalize source image coordinates to [-1, 1] and RGBs to [0, 1].
    image_norm = normalize_image(image)

    # Save pixels as data points.
    width, height = image.size
    data = dissassemble_image_chunk(
        width, height, image_norm, spp=spp, to_stratify=False)

    return data


def prepare_pickled_data(sample_pickle):
    # The pickled samples are already in [0, 1]
    # The sample RGBs are already in [0, 1].
    with open(sample_pickle, 'rb') as file:
        # Load the object from the pickle file
        samples = pickle.load(file)

    data = []

    # Change coordinate to y, x
    data.append(torch.from_numpy(samples[0][:, [1, 0]]))
    data.append(torch.from_numpy(samples[1]))

    return data


class Nerf2dDataset(Dataset):
    def __init__(self, image, to_stratify=True):
        self.data = []

        # Normalize source image coordinates to [-1, 1] and RGBs to [0, 1].
        image_norm = normalize_image(image)

        # Save pixels as data points.
        width, height = image.size
        # self.data = dissassemble_image(
        #     width, height, image_norm, spp=10, to_stratify=True)
        self.data = dissassemble_image(
            width, height, image_norm, spp=1, to_stratify=to_stratify)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Nerf2dLatentDataset(Dataset):
    def __init__(self, image, pipe, upsampling_rate=1):
        self.data = []

        # Save the image that we are going to compose with
        self.image = image
        self.image_norm = torch.from_numpy(normalize_image(image))
        self.latent_image = pipe.to_latent_space(self.image_norm)

        # Upsample the original image to generate more samples
        if upsampling_rate != 1:
            upsampled_image = image.resize(
                (upsampling_rate * image.width, upsampling_rate * image.height), Image.BILINEAR)
        else:
            upsampled_image = image
        upsampled_image_norm = torch.from_numpy(
            normalize_image(upsampled_image))
        upsampled_latent_image = pipe.to_latent_space(upsampled_image_norm)

        # Save pixels as data points.
        height, width = upsampled_latent_image.shape[-2:]
        self.data = dissassemble_image(width, height, upsampled_latent_image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PseudoDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor([0, 0]), torch.tensor([0, 0, 0])
