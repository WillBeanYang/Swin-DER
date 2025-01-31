from typing import Tuple, Union, List

from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
import pywt
import torch


class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales: Union[List, Tuple],
                 order: int = 0, input_key: str = "seg",
                 output_key: str = "seg", axes: Tuple[int] = None):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        if self.axes is None:
            axes = list(range(2, len(data_dict[self.input_key].shape)))
        else:
            axes = self.axes

        output = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * len(axes)
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all([i == 1 for i in s]):
                output.append(data_dict[self.input_key])
            else:
                new_shape = np.array(data_dict[self.input_key].shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                out_seg = np.zeros(new_shape, dtype=data_dict[self.input_key].dtype)
                for b in range(data_dict[self.input_key].shape[0]):
                    for c in range(data_dict[self.input_key].shape[1]):
                        out_seg[b, c] = resize_segmentation(data_dict[self.input_key][b, c], new_shape[2:], self.order)
                output.append(out_seg)
        data_dict[self.output_key] = output
        return data_dict

class DownsampleArray(AbstractTransform):

    def __init__(self, ds_scales: Union[List, Tuple, np.ndarray],
                 order: int = 0, axes: Tuple[int] = None):
        self.axes = axes
        self.ds_scales = np.array(ds_scales) if not isinstance(ds_scales, np.ndarray) else ds_scales
        self.order = order

    def __call__(self, data):
        is_np = True if isinstance(data, np.ndarray) else False

        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if self.axes is None:
            axes = list(range(2, len(data.shape)))
        else:
            axes = self.axes

        output = []

        for s in self.ds_scales:
            if not isinstance(s, (tuple, list, np.ndarray)):
                s = [s] * len(axes)
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all([i == 1 for i in s]):
                output.append(data if is_np else torch.from_numpy(data))
            else:
                new_shape = np.array(data.shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                out_seg = np.zeros(new_shape, dtype=data.dtype)
                for b in range(data.shape[0]):
                    for c in range(data.shape[1]):
                        out_seg[b, c] = resize_segmentation(data[b, c], new_shape[2:], self.order)
                out_seg = out_seg if is_np else torch.from_numpy(out_seg)
                output.append(out_seg)

        return output
