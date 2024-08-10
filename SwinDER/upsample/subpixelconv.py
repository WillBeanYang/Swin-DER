import torch.nn as nn
import numpy as np
from typing import Optional, Sequence, Union

from monai.networks.layers.factories import Conv
from monai.utils import ensure_tuple_rep

class SubPixelConv(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            scale_factor: Union[Sequence[float], float] = 2,
            bias: bool = True,
        ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of channels of the input image.
            out_channels: number of channels of the output image. Defaults to `in_channels`.
            scale_factor: multiplier for spatial size. Has to match input size if it is a tuple. Defaults to 2.
            bias: whether to have a bias term in the conv layers. Defaults to True.
        """
        super().__init__()
        if scale_factor <= 0:
            raise ValueError(f"The `scale_factor` multiplier must be an integer greater than 0, got {scale_factor}.")
        self.scale_factor = ensure_tuple_rep(scale_factor, spatial_dims)
        self.dimensions = spatial_dims
        conv_out_channels = out_channels * np.prod(self.scale_factor)
        self.conv =  Conv[Conv.CONV, self.dimensions](
                in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )
        self.tanh = nn.Tanh()
        if self.dimensions == 2:
            self.pixelshuffle = nn.PixelShuffle(upscale_factor=self.scale_factor)
        elif self.dimensions == 3:
            self.pixelshuffle = PixelShuffle3d(scale_factor=self.scale_factor, out_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = self.pixelshuffle(x)
        return x


class PixelShuffle3d(nn.Module):
    """
    This class is a 3d version of pixelshuffle.
    """
    def __init__(self, scale_factor, out_channels):
        """
        Args:
            scale_factor: multiplier for spatial size. Has to match input size if it is a tuple.
            out_channels: number of channels of the output image.
        """
        super().__init__()
        self.scale_factor = np.array(scale_factor)
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        n = np.sum(self.scale_factor == 2)
        assert channels % (2 ** n) == 0
        nOut = self.out_channels

        out_depth = in_depth * self.scale_factor[0]
        out_height = in_height * self.scale_factor[1]
        out_width = in_width * self.scale_factor[2]

        input_view = x.contiguous().view(batch_size, nOut, self.scale_factor[0], self.scale_factor[1], self.scale_factor[2], in_depth,
                                         in_height,
                                         in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)