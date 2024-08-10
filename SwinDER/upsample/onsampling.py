import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.layers.factories import Conv
from monai.networks.blocks.convolutions import Convolution
from monai.networks.utils import pixelshuffle


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Onsampling(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            spatial_dims: int,
            mid_channels: int = 64,
            scale: int = 2,
            kernel_size_encoder: int = 3,
            dyscope: bool = False):
        super().__init__()
        self.scale = scale
        self.spatial_dims = spatial_dims

        if in_channels != out_channels:
            self.preconv = Conv[Conv.CONV, self.spatial_dims](
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
            in_channels = out_channels

        self.comp = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=out_channels,
            out_channels=mid_channels,
            kernel_size=1,
            strides=1,
            act="RELU",
            norm="INSTANCE",
        )
        self.enc = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=mid_channels,
            out_channels=(scale * 2) ** self.spatial_dims,
            kernel_size=kernel_size_encoder,
            strides=1,
            act=None,
            norm="INSTANCE",
        )

        self.offset = Conv[Conv.CONV, self.spatial_dims](in_channels=in_channels,
                                                         out_channels=self.spatial_dims * scale ** self.spatial_dims,
                                                         kernel_size=1)
        normal_init(self.offset, std=0.001)

        if dyscope:
            self.scope = Conv[Conv.CONV, self.spatial_dims](in_channels=in_channels,
                                                            out_channels=self.spatial_dims * scale ** self.spatial_dims,
                                                            kernel_size=1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        if self.spatial_dims == 2:
            return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).reshape(1, -1, 1, 1)
        elif self.spatial_dims == 3:
            return torch.stack(torch.meshgrid([h, h, h])).transpose(1, 3).reshape(1, -1, 1, 1, 1)

    def pixelshuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel shuffle to the tensor `x`, moving pixels from the channel dimension to spatial dimensions.

        See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
        Using a nEfficient Sub-Pixel Convolutional Neural Network."

        See: Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

        Args:
            x: Input tensor

        Returns:
            Reshuffled version of `x`.

        Raises:
            ValueError: When spatial dimensions and coordinate dimensions do not match
            ValueError: When input channels of `x` are not divisible by (scale_factor ** spatial_dims)
        """
        dim, factor = self.spatial_dims, self.scale
        input_size = list(x.size())
        keeped_dim = input_size[:-(dim + 1)]
        channels = input_size[-(dim + 1)]

        if len(keeped_dim) == 2 and dim != keeped_dim[1]:
            raise ValueError(
                f"The data has a dimension of {dim}, while the coordinate dimension is {keeped_dim[1]}."
            )
        scale_divisor = factor ** dim
        if channels % scale_divisor != 0:
            raise ValueError(
                f"Number of input channels ({channels}) must be evenly "
                f"divisible by scale_factor ** dimensions ({factor}**{dim}={scale_divisor})."
            )

        spatial_start_idx = len(keeped_dim) + 1
        org_channels = int(channels // scale_divisor)
        output_size = keeped_dim + [org_channels] + [d * factor for d in input_size[spatial_start_idx:]]

        indices = list(range(spatial_start_idx, spatial_start_idx + 2 * dim))
        indices = indices[dim:] + indices[:dim]
        permute_indices = list(range(spatial_start_idx))
        for idx in range(dim):
            permute_indices.extend(indices[idx::dim])

        x = x.reshape(keeped_dim + [org_channels] + [factor] * dim + input_size[spatial_start_idx:])
        x = x.permute(permute_indices).reshape(output_size)
        return x

    def get_grid(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos

        if self.spatial_dims == 2:
            B, _, H, W = offset.shape
            offset = offset.view(B, 2, -1, H, W)
            coords_h = torch.arange(H) + 0.5
            coords_w = torch.arange(W) + 0.5
            coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                                 ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
            grid = self.pixelshuffle(coords + offset).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
            return grid
        elif self.spatial_dims == 3:
            B, _, D, H, W = offset.shape
            offset = offset.view(B, 3, -1, D, H, W)
            coords_d = torch.arange(D) + 0.5
            coords_h = torch.arange(H) + 0.5
            coords_w = torch.arange(W) + 0.5
            coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])
                                 ).transpose(1, 3).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
            grid = self.pixelshuffle(coords + offset).permute(0, 2, 3, 4, 5, 1).contiguous().flatten(0, 1)
            return grid

    def get_neighbor_pixels(self, x, grid):
        if self.spatial_dims == 2:
            B, C, H, W = x.shape
            _, h_, w_, _ = grid.shape
            offsets = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], device=grid.device)
            coords = (grid.unsqueeze(0) + offsets.unsqueeze(1).unsqueeze(2)).view(-1, h_, w_, 3)
            normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 1, 1, 3)
            coords = 2 * coords / normalizer - 1
            X = F.grid_sample(x.repeat(8, 1, 1, 1), coords, mode='bilinear',
                              align_corners=False, padding_mode="border").view(B, 8, C, h_, w_).permute(0, 2, 1, 3, 4)
        elif self.spatial_dims == 3:
            B, C, D, H, W = x.shape
            _, d_, h_, w_ , _ = grid.shape
            offsets = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                                    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], device=grid.device)
            coords = (grid.unsqueeze(1) + offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)).view(-1, d_, h_, w_, 3)
            normalizer = torch.tensor([W, H, D], dtype=x.dtype, device=x.device).view(1, 1, 1, 1, 3)
            coords = 2 * coords / normalizer - 1
            X = F.grid_sample(x.repeat(8, 1, 1, 1, 1), coords, mode='bilinear',
                              align_corners=False, padding_mode="border").view(B, 8, C, d_, h_, w_).permute(0, 2, 1, 3, 4, 5)
        return X

    def forward(self, X):
        if hasattr(self, 'preconv'):
            X = self.preconv(X)
        W = self.comp(X)
        W = self.enc(W)
        W = pixelshuffle(W, spatial_dims=self.spatial_dims, scale_factor=self.scale)
        W = F.softmax(W, dim=1)

        grid = self.get_grid(X)
        X = self.get_neighbor_pixels(X, grid)

        if self.spatial_dims == 2:
            X = torch.einsum('bkhw,bckhw->bchw', [W, X])
        elif self.spatial_dims == 3:
            X = torch.einsum('bkdhw,bckdhw->bcdhw', [W, X])

        return X

if __name__ == '__main__':
    x = torch.rand(2, 2, 5, 5, 5).to('cuda:0')
    transdefconv = Onsampling(spatial_dims=3,in_channels=2,out_channels=1).to('cuda:0')
    oup = transdefconv(x)
    print(oup.size())
