from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn
from torchaudio import transforms

from igoose import data_map as dm


class Log10MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        win_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[Mapping[str, Any]] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        log_input_clamp_min: float = 1e-10,
        output_loudness_key: str = None,
        log_loudness_clamp_min: float = 1e-10,
    ):
        super().__init__()

        self._log_input_clamp_min = log_input_clamp_min
        self._output_loudness_key = output_loudness_key
        self._log_loudness_clamp_min = log_loudness_clamp_min

        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        input_signal = batch[dm.SIGNAL]

        mel = self.mel_spectrogram(input_signal)

        output_log_mel_spectrogram = mel.clamp(min=self._log_input_clamp_min).log10()

        output_map = {dm.CODE: output_log_mel_spectrogram}

        loudness_key = self._output_loudness_key
        if loudness_key is not None:
            loudness = mel.mean(dim=-2).clamp_min(self._log_loudness_clamp_min).log10()
            output_map[loudness_key] = loudness

        input_length = batch.get(dm.get_length_key(dm.SIGNAL))
        if input_length is not None:
            output_length = (
                input_length - self.mel_spectrogram.n_fft % 1
            ) // self.mel_spectrogram.hop_length + 1

            output_map[dm.get_length_key(dm.CODE)] = output_length

            if loudness_key is not None:
                output_map[dm.get_length_key(loudness_key)] = output_length

        return output_map
