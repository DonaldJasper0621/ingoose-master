import functools
import logging
from typing import Any, Iterable, Mapping, Optional

import numpy as np
from scipy import signal
import torch

from igoose import data_map as dm
from igoose.nn import resamplings
from igoose.ops import array_ops_np
from igoose.ops import signal_ops_np
from igoose.transforms import signal_transforms_np

log = logging.getLogger(__name__)


AtMostLength = signal_transforms_np.AtMostLengthNp


class ComputeF0Parselmouth:
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        f0_floor: float,
        f0_ceiling: float,
        silence_threshold: float = 0.03,
        voicing_threshold: float = 0.45,
        very_accurate: bool = False,
        signal_key: str = dm.SIGNAL,
        f0_key: str = dm.F0,
    ):
        self._sample_rate = sample_rate
        self._hop_length = hop_length
        self._f0_floor = f0_floor
        self._f0_ceiling = f0_ceiling
        self._silence_threshold = silence_threshold
        self._voicing_threshold = voicing_threshold
        self._very_accurate = very_accurate
        self._signal_key = signal_key
        self._f0_key = f0_key

    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        x = data_map[self._signal_key]
        single_channel_x_np = x.numpy().squeeze()
        if single_channel_x_np.ndim != 1:
            raise NotImplementedError

        single_channel_f0_np = signal_ops_np.compute_f0_parselmouth(
            single_channel_x_np,
            self._sample_rate,
            self._hop_length,
            self._f0_floor,
            self._f0_ceiling,
            silence_threshold=self._silence_threshold,
            voicing_threshold=self._voicing_threshold,
            very_accurate=self._very_accurate,
        )

        if x.size(dim=-1) != x.numel():
            raise NotImplementedError

        f0 = torch.from_numpy(single_channel_f0_np).float().view(*x.size()[:-1], -1)

        return {**data_map, self._f0_key: f0}


class PadToAtLeastLength(signal_transforms_np.PadToAtLeastLengthNp):
    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        data_map_np = dict(data_map)
        for key in self._signal_keys:
            data_map_np[key] = data_map[key].numpy()

        output_data_map_np = super().__call__(data_map_np)

        output_data_map = dict(output_data_map_np)
        for key in self._signal_keys:
            output_data_map[key] = torch.from_numpy(output_data_map_np[key])

        return output_data_map


class RandomIIRFiltering:
    def __init__(
        self,
        order_min: int,
        order_max: int,
        b_a_abs_max: float,
        keep_max_magnitude: bool = False,
        probability: float = 1.0,
        signal_keys: Iterable[str] = (dm.SIGNAL,),
    ):
        self._order_min = order_min
        self._order_max = order_max
        self._b_a_abs_max = b_a_abs_max
        self._keep_max_magnitude = keep_max_magnitude
        self._probability = probability
        self._signal_keys = sorted(set(signal_keys))

    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        output_data_map = dict(data_map)

        if torch.rand(()) >= self._probability:
            return output_data_map

        xs = [data_map[key] for key in self._signal_keys]
        while True:
            order = torch.randint(self._order_min, self._order_max + 1, size=()).item()
            b_a_abs_max = self._b_a_abs_max
            b = torch.empty((order + 1,)).uniform_(-b_a_abs_max, b_a_abs_max).numpy()
            b[0] = 1.0
            a = torch.empty((order + 1,)).uniform_(-b_a_abs_max, b_a_abs_max).numpy()
            a[0] = 1.0
            ys = [
                array_ops_np.as_tensor(signal.filtfilt(b, a, x.numpy())).to(x.dtype)
                for x in xs
            ]

            for y in ys:
                if not y.isfinite().all():
                    # pylint: disable-next=logging-fstring-interpolation
                    log.warning(
                        f"Applying order-{order} filtering with `a` {a.tolist()} and "
                        f"`b` {b.tolist()} on signals with lengths {y.shape[-1]} gets "
                        "invalid output values."
                    )
                    break
            else:
                break

        if self._keep_max_magnitude:
            normalized_ys = []
            for x, y in zip(xs, ys):
                x_max_magnitude = x.abs().max()
                y_max_magnitude = y.abs().max()
                scale = (x_max_magnitude / y_max_magnitude).nan_to_num(
                    nan=0.0, posinf=0.0, neginf=0.0
                )
                normalized_ys.append(y * scale)
            ys = normalized_ys

        output_data_map.update(zip(self._signal_keys, ys))

        return output_data_map


class RandomMagnitude:
    def __init__(
        self,
        magnitude_min: float,
        magnitude_max: float,
        eps: float = 1e-8,
        signal_key: str = dm.SIGNAL,
    ):
        self._magnitude_min = magnitude_min
        self._magnitude_max = magnitude_max
        self._eps = eps
        self._signal_key = signal_key

    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        x = data_map[self._signal_key]

        x_max_magnitude = x.abs().max()

        y_max_magnitude = torch.empty(()).uniform_(
            self._magnitude_min, self._magnitude_max
        )
        y = x * (y_max_magnitude / (x_max_magnitude + self._eps))

        return {**data_map, self._signal_key: y}


class RandomShiftF0:
    def __init__(
        self,
        sample_rate: int,
        num_steps_min: float = -12.0,
        num_steps_max: float = 12.0,
        num_steps_per_octave: float = 12.0,
        signal_key: str = dm.SIGNAL,
        f0_key_and_f0_hop_length_pair: Optional[tuple[str, int]] = None,
        resampling_common_divisor: int = 100,
    ):
        self._sample_rate = sample_rate
        self._num_steps_min = num_steps_min
        self._num_steps_max = num_steps_max
        self._num_steps_per_octave = num_steps_per_octave
        self._signal_key = signal_key
        self._resampling_common_divisor = resampling_common_divisor

        self._f0_key = None
        self._f0_hop_length = None
        if f0_key_and_f0_hop_length_pair:
            self._f0_key, self._f0_hop_length = f0_key_and_f0_hop_length_pair

        self._get_resample_module = functools.cache(self._get_resample_module)

    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        x = data_map[self._signal_key]

        num_steps = torch.empty(()).uniform_(self._num_steps_min, self._num_steps_max)
        num_steps = num_steps.item()

        orig_freq = round(
            (2.0 ** (num_steps / self._num_steps_per_octave)) * self._sample_rate
        )
        orig_freq = orig_freq - (orig_freq % self._resampling_common_divisor)

        y = self._get_resample_module(orig_freq, self._sample_rate)(x)

        output_data_map = dict(data_map)
        output_data_map[self._signal_key] = y

        if self._f0_key is not None:
            x_f0 = output_data_map[self._f0_key]
            y_f0_length = self._compute_f0_length(y.size(dim=-1))

            scale = orig_freq / self._sample_rate

            if x_f0.ndim != 2 and x_f0.size(dim=0) != 1:
                raise NotImplementedError

            y_f0 = (
                torch.from_numpy(
                    np.interp(
                        np.arange(y_f0_length) * scale,
                        np.arange(x_f0.size(dim=-1)),
                        x_f0.numpy().squeeze(axis=0) * scale,
                    )
                )
                .float()
                .unsqueeze(dim=0)
            )

            output_data_map[self._f0_key] = y_f0

        return output_data_map

    def _compute_f0_length(self, audio_length):
        # We use the output length of ``librosa.stft`` with even ``n_fft`` and
        # ``center=True`` as the output length of f0.
        return audio_length // self._f0_hop_length + 1

    # pylint: disable-next=method-hidden
    def _get_resample_module(self, orig_freq, new_freq):
        del self  # Unused.

        return resamplings.KaiserBestResample(orig_freq=orig_freq, new_freq=new_freq)
