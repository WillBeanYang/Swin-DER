from typing import Tuple, Union, List
import numpy as np
import pywt
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class WaveletDownsampleArray(AbstractTransform):
    def __init__(self, ds_scales: Union[List, Tuple, np.ndarray],
                 wavelet: str = 'haar',
                 axes: Tuple[int] = None,
                 mode: str = 'approximate'):
        self.axes = axes
        self.ds_scales = np.array(ds_scales) if not isinstance(ds_scales, np.ndarray) else ds_scales
        self.wavelet = wavelet
        self.mode = mode  # mode=0 only use approximate mode =1 use all
        self.dwt_dimensions = []
        for index, scale in enumerate(self.ds_scales):
            tmp_times_now = 1 / scale  # [1,2,2]
            tmp_times_before = np.ones_like(scale)  # [1,1,1]
            if index != 0:
                tmp_times_before = 1 / self.ds_scales[index - 1]
            need_downsample = tmp_times_now != tmp_times_before
            dimension = np.sum(need_downsample)
            if dimension == 2:
                assert not need_downsample[0], "check out downsample scales, It shouldn't be {0}!!!".format(
                    need_downsample)
            self.dwt_dimensions.append(dimension)

        self.approximate = ['a' * d for d in self.dwt_dimensions]

        self.dwt_func = pywt.dwtn

    def __call__(self, data):
        is_np = True if isinstance(data, np.ndarray) else False
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if self.axes is None:
            axes = list(range(2, len(data.shape)))
        else:
            axes = self.axes
        output = []
        approximate = []
        for index, s in enumerate(self.ds_scales):
            if not isinstance(s, (tuple, list, np.ndarray)):
                s = [s] * len(axes)
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all([i == 1 for i in s]):
                output.append(data if is_np else torch.from_numpy(data))
                approximate.append(data)
            else:
                new_shape = np.array(data.shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                if self.dwt_dimensions[index] == 2:
                    transform_axes = [len(new_shape) - 2, len(new_shape) - 1]  # if 4 [2,3] if 5 [3,4]
                    if index == 0:
                        out_seg = \
                            self.dwt_func(data, self.wavelet, axes=transform_axes)
                    else:
                        out_seg = \
                            self.dwt_func(approximate[index - 1], self.wavelet, axes=transform_axes)
                else:  # self.dwt_dimensions[index] == 3
                    transform_axes = [len(new_shape) - 3, len(new_shape) - 2, len(new_shape) - 1]  # [2,3,4]
                    if index == 0:
                        out_seg = self.dwt_func(data, self.wavelet, axes=transform_axes)
                    else:
                        out_seg = self.dwt_func(approximate[index - 1], self.wavelet, axes=transform_axes)
                approximate.append(out_seg[self.approximate[index]])
                assert (out_seg[self.approximate[index]].shape == new_shape).all(), \
                    "Some error occurred new shape should be {0}," \
                    "now out seg shape is {1}".format(new_shape, out_seg[self.approximate[index]].shape)
                if self.mode == 'approximate':
                    out_seg = out_seg[self.approximate[index]]
                elif self.mode == 'all':
                    out_seg = np.concatenate(list(out_seg.values()), axis=1) # channel-wise concatenate
                elif self.mode == 'highFrequency':
                    out_seg.pop(self.approximate[index])
                    out_seg = np.concatenate(list(out_seg.values()), axis=1)
                out_seg = out_seg if is_np else torch.from_numpy(out_seg)
                output.append(out_seg)
        return output


class WaveletDownsampleD(AbstractTransform):
    """
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    """

    # TODO: only support harr and bior1.1 now
    # TODO: do not support output with Discontinuous downsampling
    def __init__(self, ds_scales: Union[List, Tuple],
                 input_key: str = "seg",
                 output_key: str = "seg",
                 wavelet: str = "haar",
                 mode: str = 'approximate',
                 axes: Tuple[int] = None):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.ds_scales = np.array(ds_scales)
        self.wavelet = wavelet
        self.mode = mode

    def __call__(self, **data_dict):
        waveletdownsample_array = WaveletDownsampleArray(ds_scales=self.ds_scales, axes=self.axes,
                                                         wavelet=self.wavelet, mode=self.mode)
        output = waveletdownsample_array(data_dict[self.input_key])
        data_dict[self.output_key] = output
        return data_dict