from typing import Any, Mapping, Optional

import librosa
import numpy as np
import torch
from torch import nn

from igoose import data_map as dm
from igoose.ops import signal_ops
from igoose.ops import signal_ops_np


class LibrosaF0Estimator(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        f0_floor: float,
        f0_ceiling: float,
        frame_length: int = 2048,
        input_magnitude: Optional[float] = None,
        input_key: str = dm.SIGNAL,
        output_key: str = dm.F0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_floor = f0_floor
        self.f0_ceiling = f0_ceiling
        self.frame_length = frame_length
        self.input_magnitude = input_magnitude
        self._input_key = input_key
        self._output_key = output_key

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # x: ``(N, C, T)``.
        x = batch[self._input_key]
        x_length = batch.get(dm.get_length_key(self._input_key))

        if x_length is not None:
            raise NotImplementedError

        if self.input_magnitude is not None:
            x, _ = signal_ops.safe_normalize_magnitude(x, self.input_magnitude)

        f0s_np = []
        for x_np_1d in x.cpu().double().flatten(end_dim=-2).numpy():
            f0s_np.append(
                librosa.pyin(
                    x_np_1d,
                    fmin=self.f0_floor,
                    fmax=self.f0_ceiling,
                    sr=self.sample_rate,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                    fill_na=0.0,
                    center=True,
                    pad_mode="constant",
                )[0]
            )

        n, c, _ = x.size()
        f0_2d = torch.from_numpy(np.stack(f0s_np))
        f0_2d = f0_2d.to(device=x.device, dtype=torch.float)
        f0 = f0_2d.view(n, c, -1)

        return {self._output_key: f0}


class ParselmouthF0Estimator(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        f0_floor: float,
        f0_ceiling: float,
        silence_threshold: float = 0.03,
        voicing_threshold: float = 0.45,
        very_accurate: bool = False,
        input_key: str = dm.SIGNAL,
        output_key: str = dm.F0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_floor = f0_floor
        self.f0_ceiling = f0_ceiling
        self.silence_threshold = silence_threshold
        self.voicing_threshold = voicing_threshold
        self.very_accurate = very_accurate
        self._input_key = input_key
        self._output_key = output_key

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # x: ``(N, C, T)``.
        x = batch[self._input_key]
        x_length = batch.get(dm.get_length_key(self._input_key))

        if x_length is not None:
            raise NotImplementedError

        f0s_np = []
        for x_np_1d in x.cpu().double().flatten(end_dim=-2).numpy():
            f0s_np.append(
                signal_ops_np.compute_f0_parselmouth(
                    x_np_1d,
                    self.sample_rate,
                    self.hop_length,
                    self.f0_floor,
                    self.f0_ceiling,
                    silence_threshold=self.silence_threshold,
                    voicing_threshold=self.voicing_threshold,
                    very_accurate=self.very_accurate,
                )
            )

        n, c, _ = x.size()
        f0_2d = torch.from_numpy(np.stack(f0s_np))
        f0_2d = f0_2d.to(device=x.device, dtype=torch.float)
        f0 = f0_2d.view(n, c, -1)

        return {self._output_key: f0}
