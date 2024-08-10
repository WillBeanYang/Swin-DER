from typing import Sequence, Union, Tuple, Optional
import math
import torch
import torch.nn as nn
import numpy as np
from dconv_native import deform_conv3d
from monai.networks.layers.factories import Conv

class DeformableConvV2(nn.Module):
    """
    The 2D/3D implementation of the Deformable Convolution V2 module.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride. Default: 1.
        padding: implicit paddings on both sides of the input. Default: None.
        dilation: the spacing between kernel elements. Default: 1.
        groups: split input into groups, in_channels should be divisible by the number of groups. Default: 1.
        deformable_groups: splitting input channels into groups, where channels within the same group share the same spatial positional offsets,
        in_channels should be divisible by the number of deformable groups. Default: 1.
    """
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int] = 1,
            padding: Optional[Union[Sequence[int], int]] = None,
            dilation: Union[Sequence[int], int] = 1,
            groups: int = 1,
            deformable_groups: int = 1):
        super(DeformableConvV2, self).__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, (list, tuple)) and len(kernel_size) == self.spatial_dims:
            self.kernel_size = tuple(kernel_size)
        elif isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * self.spatial_dims
        else:
            ValueError("feature size and kernel size mismatch!")

        if isinstance(stride, (list, tuple)) and len(stride) == self.spatial_dims:
            self.stride = tuple(stride)
        elif isinstance(stride, int):
            self.stride = (stride,) * self.spatial_dims
        else:
            ValueError("feature size and stride size mismatch!")

        if isinstance(padding, (list, tuple)) and len(padding) == self.spatial_dims:
            self.padding = tuple(padding)
        elif isinstance(padding, int):
            self.padding = (padding,) * self.spatial_dims
        elif padding == None:
            self.padding = self.get_padding(self.kernel_size, self.stride)
        else:
            ValueError("feature size and padding size mismatch!")

        if isinstance(dilation, (list, tuple)) and len(dilation) == self.spatial_dims:
            self.dilation = tuple(dilation)
        elif isinstance(dilation, int):
            self.dilation = (dilation,) * self.spatial_dims
        else:
            ValueError("feature size and dilation size mismatch!")

        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * (self.spatial_dims + 1) * np.prod(self.kernel_size))
        self.conv_offset_mask = Conv[Conv.CONV, self.spatial_dims](
            in_channels=self.in_channels,
            out_channels=out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.reset_parameters()

    def get_padding(
            self, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
    ) -> Union[Tuple[int, ...], int]:
        kernel_size_np = np.atleast_1d(kernel_size)
        stride_np = np.atleast_1d(stride)
        padding_np = (kernel_size_np - stride_np + 1) / 2
        if np.min(padding_np) < 0:
            raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
        padding = tuple(int(p) for p in padding_np)

        return padding if len(padding) > 1 else padding[0]

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        output = torch.chunk(offset_mask, self.spatial_dims + 1, dim=1)
        offset = torch.cat(output[:-1], dim=1)
        mask = torch.sigmoid(output[-1])
        if self.spatial_dims == 2:
            x = torch.ops.torchvision.deform_conv2d(
                x,
                self.weight,
                offset,
                mask,
                self.bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1],
                self.groups,
                self.deformable_groups,
                True
            )
        elif self.spatial_dims == 3:
            x = deform_conv3d(
                input=x,
                offset=offset,
                alpha=mask,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                n_weight_groups=self.groups,
                n_offset_groups=self.deformable_groups,
            )
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


if __name__ == '__main__':
    x = torch.rand(1, 2, 5, 5).to('cuda')
    dcnv2 = DeformableConvV2(spatial_dims=2, in_channels=2, out_channels=3, kernel_size=3).to('cuda')
    y = dcnv2(x)
    print(y.shape)
