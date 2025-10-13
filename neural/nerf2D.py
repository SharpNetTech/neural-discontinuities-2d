import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from learning.nerf2d_data import normalize_image
from tools.math_utils import rotation_matrix
from tools.plot_utils import to_pil_image
from tools.utils import learning_rate_decay, compose_image


def positional_encoding(coord, encoding_type, L=10):
    if encoding_type == 'none':
        return coord

    coord_el = []
    for el in range(0, L):
        val = 2 ** el
        if encoding_type == 'sin_cos':
            x = torch.sin(val * math.pi * coord[:, 0])
            coord_el.append(x.view(-1, 1))

            x = torch.cos(val * math.pi * coord[:, 0])
            coord_el.append(x.view(-1, 1))

            y = torch.sin(val * math.pi * coord[:, 1])
            coord_el.append(y.view(-1, 1))

            y = torch.cos(val * math.pi * coord[:, 1])
            coord_el.append(y.view(-1, 1))

    if encoding_type == 'sin_cos':
        coord_els = torch.hstack(coord_el)
        return torch.cat((coord, coord_els), dim=1).clone().detach().float()

    assert False, 'Invalid encoding type'


class Sine(nn.Module):
    """Simple sine activation used when opt_config['activation'] == 'sine'.
    The original implementation lived in nerf1D; we inline a minimal version here
    to avoid that dependency.
    """
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class MLP(pl.LightningModule):
    def __init__(self, image, layer_dims, opt_config, encoding_type='none', L=10):
        super(MLP, self).__init__()

        # Nerf parameters
        self.encoding_type = encoding_type
        self.L = L

        # Correctly set input layer dimension
        if L >= 0:
            layer_dims[0] = positional_encoding(
                torch.tensor([[0, 0]]), encoding_type, L).shape[1]

        layers = []
        for i, dim in enumerate(layer_dims):
            if i > 1:
                # layers.append(nn.ReLU())
                # layers.append(Sine())
                if opt_config['activation'] == 'tanh':
                    layers.append(nn.Tanh())
                elif opt_config['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif opt_config['activation'] == 'sine':
                    layers.append(Sine())
                else:
                    assert False, 'Invalid activation function'
            if i > 0:
                layers.append(nn.Linear(layer_dims[i-1], dim))
                nn.init.xavier_normal_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)

        # layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        self.image = image
        self.image_norm = torch.from_numpy(normalize_image(self.image))
        self.image_dim = image.size
        self.opt_config = opt_config

        self.fit_latent = False

    def forward(self, x):
        output_dim = self.layers[-1].out_features

        # Apply sigmoid to the alpha channel if it exists
        if output_dim == 3 or output_dim == 1:
            return self.layers(x)
        elif output_dim == 4 or output_dim == 5:
            v = self.layers(x)
            if not self.fit_latent:
                v[:, output_dim-1] = torch.sigmoid(v[:, output_dim-1])
                # v = torch.sigmoid(v)
                # v = torch.clamp(v, min=-3, max=3)

            return v

    def configure_optimizers(self):
        if 'learning_rate' in self.opt_config:
            return [optim.Adam(self.parameters(),
                               lr=self.opt_config['learning_rate'],
                               betas=(0.9, 0.999) if 'betas' not in self.opt_config else self.opt_config['betas'])], \
                None
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.opt_config['lr_init'],
                betas=(
                    0.9, 0.999) if 'betas' not in self.opt_config else self.opt_config['betas'],
                eps=1e-6)

            lr_init = self.opt_config['lr_init']
            lr_final = self.opt_config['lr_final']
            lr_delay_steps = self.opt_config['lr_delay_steps']
            lr_delay_mult = self.opt_config['lr_delay_mult']

            num_epochs = self.opt_config['num_epochs']

            def lr_lambda(step): return learning_rate_decay(step, lr_init, lr_final, num_epochs,
                                                            lr_delay_steps=lr_delay_steps,
                                                            lr_delay_mult=lr_delay_mult) / lr_init

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda, last_epoch=-1)

            return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        encoded_x = positional_encoding(x, self.encoding_type, self.L)
        encoded_x = encoded_x.to(self.device)
        y_hat = self(encoded_x)
        loss_func = nn.MSELoss()
        loss = loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output_dim = self.layers[-1].out_features
        assert output_dim == 3, 'Not fitting to the image'

        # x, y = batch
        # encoded_x = positional_encoding(x, self.encoding_type, self.L)
        # encoded_x = encoded_x.to(self.device)
        # y_hat = self(encoded_x)

        # Run the validation on the entire image since we are overfitting
        width, height = self.image_dim
        y_coords, x_coords = torch.meshgrid(torch.arange(
            height), torch.arange(width), indexing='ij')
        y_coords = (y_coords.float() + 0.5) / height
        x_coords = (x_coords.float() + 0.5) / width
        coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)
        encoded_coord = positional_encoding(
            coords, self.encoding_type, self.L)
        encoded_coord = encoded_coord.to(self.device)
        colors_mlp = self(encoded_coord)

        image_norm = normalize_image(self.image)
        width, height = self.image.size
        colors = []
        for y in range(height):
            for x in range(width):
                r, g, b = image_norm[y, x]
                colors.append(torch.tensor([r, g, b]))
        colors = torch.vstack(colors).to(self.device)

        loss_func = nn.MSELoss()
        loss = loss_func(colors_mlp, colors)
        self.log('val_loss', loss, sync_dist=True)

        return loss

    def evaluate(self, zoom=1.0, viewbox=None):
        raw_px = self.evaluate_raw(zoom=zoom, viewbox=viewbox)
        raw_px = raw_px.cpu()

        # Alpha blending
        output_dim = self.layers[-1].out_features
        if output_dim == 4:
            raw_px = compose_image(self.image_norm, raw_px)

        image = to_pil_image(raw_px)

        return image

    def coord_transform(self, coord, angle=45.0):
        if angle == 0:
            return coord

        # Rotate the coordinate by given degrees
        coord = rotation_matrix(angle) @ coord
        return coord

    def evaluate_raw(self, zoom=1.0, viewbox=None, x_=None):
        width, height = self.image_dim
        width, height = int(width * zoom), int(height * zoom)
        output_w, output_h = width, height

        if isinstance(x_, torch.Tensor):
            coords = x_
        else:
            coords = []
            if viewbox:
                x, y, x2, y2 = viewbox
                w = x2 - x
                h = y2 - y

                x, y, w, h = x * zoom, y * zoom, w * zoom, h * zoom
                # print(f'x: {x}, y: {y}, w: {w}, h: {h}')
                if w > h:
                    output_w = int(width)
                    output_h = int(math.ceil(h * width / w))
                else:
                    output_h = int(height)
                    output_w = int(math.ceil(w * height / h))
                # print(f'output_w: {output_w}, output_h: {output_h}')
                y_coords, x_coords = torch.meshgrid(torch.linspace(
                    y, y + h, output_h), torch.linspace(x, x + w, output_w), indexing='ij')
                zoom_x = output_w / w
                zoom_y = output_h / h
                y_coords = (y_coords.float() + 0.5 / zoom_y)
                x_coords = (x_coords.float() + 0.5 / zoom_x)
                # y_coords = (y_coords.float() + 0.5)
                # x_coords = (x_coords.float() + 0.5)
            else:
                y_coords, x_coords = torch.meshgrid(torch.arange(
                    height), torch.arange(width), indexing='ij')
                y_coords = (y_coords.float() + 0.5)
                x_coords = (x_coords.float() + 0.5)

            y_coords = y_coords / height
            x_coords = x_coords / width

            coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)

        encoded_coord = positional_encoding(
            coords, self.encoding_type, self.L)
        encoded_coord = encoded_coord.to(self.device)
        encoded_coord.requires_grad = False
        raw_pixel = self(encoded_coord)

        output_dim = self.layers[-1].out_features
        raw_pixel = raw_pixel.reshape(output_h, output_w, output_dim)

        return raw_pixel
