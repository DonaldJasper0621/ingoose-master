from typing import Any

from numpy import typing as npt
from scipy.io import wavfile


def read_wav(path, mmap: bool = False) -> tuple[int, npt.NDArray[Any]]:
    sample_rate, samples = wavfile.read(path, mmap=mmap)
    if samples.ndim == 2:
        return sample_rate, samples.T

    if samples.ndim == 1:
        return sample_rate, samples[None]

    raise NotImplementedError
