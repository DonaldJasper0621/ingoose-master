import numpy as np
import torch


def _get_writable_array_with_positive_strides(array):
    if (not array.flags.writeable) or any(stride < 0 for stride in array.strides):
        array = array.copy()
    return array


def as_audio_tensor(samples):
    samples = np.asarray(samples)
    if samples.dtype == np.int16:
        return torch.from_numpy(samples.astype(np.float32) / 32768.0)

    if samples.dtype == np.int32:
        return torch.from_numpy(samples.astype(np.float32) / 2147483648.0)

    if samples.dtype != np.float32:
        raise NotImplementedError

    return torch.from_numpy(_get_writable_array_with_positive_strides(samples))


def as_tensor(array):
    return torch.from_numpy(_get_writable_array_with_positive_strides(array))
