from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms

from igoose import data_map as dm


class MultiResolutionLogMelL1Loss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft_and_win_length_and_hop_length_and_n_mels_tuples: Sequence[
            tuple[int, int, int, int]
        ],
        loss_scale: float = 1.0,
        normalized_loss_by_num_mels: bool = True,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        pad_mode: str = "reflect",
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        log_input_clamp_min: float = 1e-5,
        prediction_key: str = dm.SIGNAL,
        target_signal_key: str = dm.TARGET_SIGNAL,
    ):
        super().__init__()

        self.name_to_mel_spectrogram = nn.ModuleDict(
            modules={
                f"mel_{n_fft}_{win_length}_{hop_length}_{n_mels}": (
                    transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        f_min=f_min,
                        f_max=f_max,
                        n_mels=n_mels,
                        power=power,
                        pad_mode=pad_mode,
                        norm=norm,
                        mel_scale=mel_scale,
                    )
                )
                for n_fft, win_length, hop_length, n_mels in (
                    n_fft_and_win_length_and_hop_length_and_n_mels_tuples
                )
            }
        )

        self._loss_scale = loss_scale
        if normalized_loss_by_num_mels:
            self._loss_scale /= len(self.name_to_mel_spectrogram)

        self._log_input_clamp_min = log_input_clamp_min
        self._prediction_key = prediction_key
        self._target_signal_key = target_signal_key

    def forward(
        self, output_map: Mapping[str, Any], data_map: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        output_signal = output_map[self._prediction_key]
        target_signal = data_map[self._target_signal_key]

        loss_map = {}
        for name, mel_spectrogram in self.name_to_mel_spectrogram.items():
            output_mel = mel_spectrogram(output_signal)
            target_mel = mel_spectrogram(target_signal)
            loss_map[f"log_mel_l1_loss/{name}"] = (
                F.l1_loss(
                    output_mel.clamp(min=self._log_input_clamp_min).log(),
                    target_mel.clamp(min=self._log_input_clamp_min).log(),
                )
                * self._loss_scale
            )

        return loss_map


class MultiResolutionSTFTMagnitudeL1Loss(nn.Module):
    def __init__(
        self,
        n_fft_and_win_length_and_hop_length_tuples: Sequence[tuple[int, int, int]] = (
            (512, 240, 50),
            (1024, 600, 120),
            (2048, 1120, 240),
        ),
        loss_scale: float = 1.0,
        normalized_loss_by_num_stfts: bool = True,
        prediction_key: str = dm.SIGNAL,
        target_signal_key: str = dm.TARGET_SIGNAL,
    ):
        super().__init__()

        self.name_to_stft = nn.ModuleDict(
            modules={
                f"stf_{n_fft}_{win_length}_{hop_length}": transforms.Spectrogram(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    power=None,
                )
                for n_fft, win_length, hop_length in (
                    n_fft_and_win_length_and_hop_length_tuples
                )
            }
        )

        self._loss_scale = loss_scale
        if normalized_loss_by_num_stfts:
            self._loss_scale /= len(self.name_to_stft)

        self._prediction_key = prediction_key
        self._target_signal_key = target_signal_key

    def forward(
        self, output_map: Mapping[str, Any], data_map: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        output_signal = output_map[self._prediction_key]
        target_signal = data_map[self._target_signal_key]

        loss_map = {}
        for name, stft in self.name_to_stft.items():
            output_magnitude = stft(output_signal).abs()
            target_magnitude = stft(target_signal).abs()
            loss_map[f"magnitude_l1_loss/{name}"] = (
                F.l1_loss(output_magnitude, target_magnitude) * self._loss_scale
            )

        return loss_map


class TimeDomainEnvelopL1Loss(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        loss_scale: float = 1.0,
        prediction_key: str = dm.SIGNAL,
        target_signal_key: str = dm.TARGET_SIGNAL,
    ):
        super().__init__()

        self.max_pool = nn.MaxPool1d(kernel_size, stride=stride)

        self._loss_scale = loss_scale
        self._prediction_key = prediction_key
        self._target_signal_key = target_signal_key

    def compute_upper_envelop(self, signal):
        signal_3d = signal.flatten(end_dim=-2)[:, None]
        return self.max_pool(signal_3d)

    def forward(
        self, output_map: Mapping[str, Any], data_map: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        output_signal = output_map[self._prediction_key]
        target_signal = data_map[self._target_signal_key]

        output_upper_envelop = self.compute_upper_envelop(output_signal)
        output_lower_envelop = self.compute_upper_envelop(-output_signal)
        target_upper_envelop = self.compute_upper_envelop(target_signal)
        target_lower_envelop = self.compute_upper_envelop(-target_signal)

        upper_envelop_loss = F.l1_loss(output_upper_envelop, target_upper_envelop)
        lower_envelop_loss = F.l1_loss(output_lower_envelop, target_lower_envelop)

        return {
            "envelop_l1_loss": (
                (upper_envelop_loss + lower_envelop_loss) * self._loss_scale
            )
        }
