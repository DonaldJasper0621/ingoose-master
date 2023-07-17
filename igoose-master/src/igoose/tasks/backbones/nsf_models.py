from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import torch
from torch import jit
from torch import nn
from torch.nn import functional as F
from torch.nn import utils as nn_utils

from igoose import data_map as dm
from igoose.nn import activations
from igoose.nn import pqmf
from igoose.ops import signal_ops
from igoose.tasks.backbones import utils
from igoose.utilities import module_tools


class ResBlock1d(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation_type: str = "leaky_relu",
        apply_activation_on_second_conv: bool = False,
    ):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(
                num_input_channels,
                num_input_channels,
                kernel_size,
                padding=kernel_size // 2 * dilation,
                dilation=dilation,
            ),
            utils.create_activation(activation_type, num_channels=num_input_channels),
            nn.Conv1d(
                num_input_channels,
                num_input_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
        )
        if apply_activation_on_second_conv:
            self.residual.append(
                utils.create_activation(
                    activation_type, num_channels=num_input_channels
                )
            )

    def forward(self, x):
        return x + self.residual(x)


class NSFResNetStageConfig(NamedTuple):
    num_channels: int
    upsampling_stride: int
    res_block_kernel_size_and_dilation_pairs: Sequence[tuple[int, int]]


class SineHarmonics(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        num_harmonics: int = 8,
        sine_amplitude=0.1,
        noise_std=0.003,
    ):
        super().__init__()

        self._sample_rate = sample_rate
        self._num_harmonics = num_harmonics
        self._sine_amplitude = sine_amplitude
        self._noise_std = noise_std

    def forward(self, f0: torch.Tensor, f0_hop_length: int = 1) -> torch.Tensor:
        """

        Args:
            f0: A tensor of shape ``(..., T)``.
            f0_hop_length:

        Returns:
            A tensor of shape ``(..., 1 + num_harmonics, T)``.
        """
        f0 = signal_ops.linear_upsample(f0, f0_hop_length)

        harmonics_f0 = f0[..., None] * torch.arange(
            1, self._num_harmonics + 2, device=f0.device
        )

        normalized_harmonics_f0 = (harmonics_f0 / self._sample_rate).frac()

        voiced_mask = f0 > 0

        if self.training:
            # Add a random phase shift to each voice segment.
            padded_voiced_mask = F.pad(voiced_mask, [1, 1])
            padded_voiced_segment_stop_mask = torch.logical_and(
                padded_voiced_mask[..., :-1], ~padded_voiced_mask[..., 1:]
            )
            uv_to_v_mask = padded_voiced_segment_stop_mask[..., :-1]
            voiced_segment_random_initial_phase_offset = (
                torch.rand_like(normalized_harmonics_f0) * uv_to_v_mask[..., None]
            )
            voiced_segment_random_initial_phase_offset[..., 0] = 0.0

            normalized_harmonics_f0 = (
                normalized_harmonics_f0 + voiced_segment_random_initial_phase_offset
            ).frac()

        normalized_harmonics_f0 = normalized_harmonics_f0.double()
        period_mask = (normalized_harmonics_f0.cumsum(dim=-2).frac()).diff(dim=-2) < 0
        period_offset = torch.zeros_like(normalized_harmonics_f0)
        period_offset[..., 1:, :] = -1.0 * period_mask
        stable_cumsum_normalized_harmonics_f0 = normalized_harmonics_f0 + period_offset

        harmonics = torch.sin(
            stable_cumsum_normalized_harmonics_f0.cumsum(-2) * (2.0 * torch.pi)
        ).to(f0.dtype)

        std = torch.where(voiced_mask, self._noise_std, self._sine_amplitude / 3.0)
        harmonics = torch.normal(
            harmonics * (voiced_mask * self._sine_amplitude)[..., None], std[..., None]
        )

        return harmonics.transpose(-2, -1)


class NSFResNet(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        f0_hop_length: int,
        num_input_channels: int,
        bottom_num_channels: int,
        stage_configs: Sequence[NSFResNetStageConfig],
        source_num_harmonics: int = 8,
        bottom_kernel_size: int = 7,
        output_kernel_size: int = 7,
        with_weight_norm: bool = True,
        activation_type: str = "leaky_relu",
        res_block_apply_activation_on_second_conv: bool = False,
        output_pqmf_kernel_size_and_stride: Optional[tuple[int, int]] = None,
    ):
        super().__init__()

        self._f0_hop_length = f0_hop_length

        self.harmonics = SineHarmonics(sample_rate, num_harmonics=source_num_harmonics)
        self.source = nn.Sequential(
            nn.Conv1d(source_num_harmonics + 1, 1, 1, bias=False),
            nn.Tanh(),
        )

        self.bottom = nn.Conv1d(
            num_input_channels,
            bottom_num_channels,
            bottom_kernel_size,
            padding=bottom_kernel_size // 2,
        )

        output_pqmf_kernel_size = None
        output_pqmf_stride = None
        if output_pqmf_kernel_size_and_stride is not None:
            (
                output_pqmf_kernel_size,
                output_pqmf_stride,
            ) = output_pqmf_kernel_size_and_stride

        self.stages_upsampling = nn.ModuleList()
        self.stages_source = nn.ModuleList()
        self.stages = nn.ModuleList()

        upsampling_strides = [
            stage_config.upsampling_stride for stage_config in stage_configs
        ]
        self.stride = np.prod(upsampling_strides).item() * (output_pqmf_stride or 1)
        stage_strides = np.cumprod([1] + upsampling_strides[:0:-1])[::-1] * (
            output_pqmf_stride or 1
        )
        prev_stage_num_channels = bottom_num_channels
        for stage_config, stage_stride in zip(stage_configs, stage_strides):
            if stage_config.upsampling_stride <= 1:
                raise NotImplementedError

            self.stages_upsampling.append(
                nn.ConvTranspose1d(
                    prev_stage_num_channels,
                    stage_config.num_channels,
                    stage_config.upsampling_stride * 2,
                    stride=stage_config.upsampling_stride,
                    padding=(stage_config.upsampling_stride + 1) // 2,
                    output_padding=stage_config.upsampling_stride % 2,
                )
            )
            self.stages_source.append(
                nn.Conv1d(
                    1,
                    stage_config.num_channels,
                    kernel_size=1 if stage_stride == 1 else stage_stride * 2,
                    stride=stage_stride,
                    padding=stage_stride - 1,
                )
            )
            self.stages.append(
                nn.Sequential(
                    *[
                        ResBlock1d(
                            stage_config.num_channels,
                            kernel_size=kernel_size,
                            dilation=dilations,
                            activation_type=activation_type,
                            apply_activation_on_second_conv=(
                                res_block_apply_activation_on_second_conv
                            ),
                        )
                        for kernel_size, dilations in (
                            stage_config.res_block_kernel_size_and_dilation_pairs
                        )
                    ]
                )
            )
            prev_stage_num_channels = stage_config.num_channels

        self.top = nn.Conv1d(
            stage_configs[-1].num_channels,
            output_pqmf_stride or 1,
            output_kernel_size,
            padding=(output_kernel_size - 1) // 2,
        )

        if with_weight_norm:
            name_to_conv = {
                name: module
                for name, module in self.named_modules()
                if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))
            }
            for name, conv in name_to_conv.items():
                if name.startswith("source."):
                    continue

                if name.startswith("stages_source."):
                    continue

                match conv:
                    case nn.Conv1d():
                        dim = 0
                    case nn.ConvTranspose1d():
                        dim = 1
                    case _:
                        raise NotImplementedError

                module_tools.set_submodule(
                    self, name, nn_utils.weight_norm(conv, dim=dim)
                )

        self.output_pqmf = None
        if output_pqmf_kernel_size_and_stride is not None:
            self.output_pqmf = pqmf.PQMF(
                output_pqmf_stride, output_pqmf_kernel_size, trainable=False
            )

        self.output_activation = activations.Clamp(
            value_min=-1.0, value_max=1.0, maybe_keep_clamped_value_gradient=True
        )

    @jit.export
    def compute_source(self, f0: torch.Tensor) -> torch.Tensor:
        """

        Args:
            f0: A tensor of shape ``(..., T)``.

        Returns:
            A tensor of shape ``(..., 1 + num_harmonics, T)``.
        """
        harmonics = self.harmonics(f0, f0_hop_length=self._f0_hop_length)
        return self.source(harmonics)

    @jit.export
    def compute_output_length(
        self, input_length: int | torch.Tensor
    ) -> int | torch.Tensor:
        return input_length * self.stride

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, source: Optional[torch.Tensor] = None
    ):
        """

        Args:
            x: A tensor of shape ``(N, C, T)``.
            f0: A tensor of shape ``(N, T)``.
            source: An optional tensor of shape ``(N, 1 + num_harmonics, T)``.

        Returns:
            A tensor of shape ``(N, T)``.
        """
        if source is None:
            source = self.compute_source(f0)

        x = self.bottom(x)

        for upsampling, stage_source, stage in zip(
            self.stages_upsampling, self.stages_source, self.stages
        ):
            x = stage(upsampling(x) + stage_source(source))

        x = self.top(x)

        if self.output_pqmf:
            x = self.output_pqmf.synthesize(x, reconstruct_boundary=False)

        return self.output_activation(x)


class NSFResNetBackbone(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        f0_hop_length: int,
        num_input_channels: int,
        bottom_num_channels: int,
        stage_configs: Sequence[NSFResNetStageConfig],
        source_num_harmonics: int = 8,
        bottom_kernel_size: int = 7,
        output_kernel_size: int = 7,
        with_weight_norm: bool = True,
        activation_type: str = "leaky_relu",
        res_block_apply_activation_on_second_conv: bool = False,
        output_pqmf_kernel_size_and_stride: Optional[tuple[int, int]] = None,
    ):
        super().__init__()

        self.network = NSFResNet(
            sample_rate,
            f0_hop_length,
            num_input_channels,
            bottom_num_channels,
            stage_configs,
            source_num_harmonics=source_num_harmonics,
            bottom_kernel_size=bottom_kernel_size,
            output_kernel_size=output_kernel_size,
            with_weight_norm=with_weight_norm,
            activation_type=activation_type,
            res_block_apply_activation_on_second_conv=(
                res_block_apply_activation_on_second_conv
            ),
            output_pqmf_kernel_size_and_stride=output_pqmf_kernel_size_and_stride,
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # input_code: ``(N, ..., T)``.
        input_code = batch[dm.CODE]
        # input_f0: ``(N, 1, T)``.
        input_f0 = batch[dm.F0]

        output_signal = self.network(
            input_code.flatten(start_dim=1, end_dim=-2), input_f0.squeeze(dim=1)
        )
        # output_signal: ``(N, 1, T)``.

        output_map = {}

        input_code_length = batch.get(dm.get_length_key(dm.CODE))
        if input_code_length is not None:
            output_signal_length = self.network.compute_output_length(input_code_length)
            output_map[dm.get_length_key(dm.SIGNAL)] = output_signal_length

        if self.training:
            target_signal = batch[dm.TARGET_SIGNAL]
            target_length = target_signal.size(dim=-1)
            output_signal = output_signal[..., :target_length]

            output_map[dm.GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR] = {
                "signal": (output_signal, target_signal)
            }

        output_map[dm.SIGNAL] = output_signal

        return output_map
