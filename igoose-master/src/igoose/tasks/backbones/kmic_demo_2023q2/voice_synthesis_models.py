import math
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from igoose import data_map as dm
from igoose.ops import grad_ops
from igoose.tasks.backbones import utils


class ResidualBlock1d(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation_type: str = "silu",
        normalization_type: str = "bn",
    ):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(
                num_in_channels,
                num_in_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2 * dilation,
                dilation=dilation,
                bias=False,
            ),
            utils.create_normalization(normalization_type, num_in_channels),
            utils.create_activation(activation_type),
            nn.Conv1d(
                num_in_channels, num_in_channels, kernel_size=3, padding=1, bias=False
            ),
            utils.create_normalization(normalization_type, num_in_channels),
        )

    def forward(self, x):
        return x + self.residual(x)


class RandomNormalNoise(nn.Module):
    def __init__(self, scale_size: tuple[int] = ()):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(scale_size))

    def forward(self, x):
        if not self.training:
            return x

        return x + torch.randn_like(x) * self.scale


class Buffer(nn.Module):
    def __init__(self, buffer):
        super().__init__()

        self.register_buffer("buffer", buffer.detach().clone())

    def forward(self, *args, **kwargs):
        del args, kwargs  # Unused.

        return self.buffer


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_conditioning_channels: int,
        num_affine_channels: int,
        activation_type: str = "silu",
        with_random_noise: bool = False,
        noise_scale_size: tuple[int] = (),
    ):
        super().__init__()
        self.conv_0 = nn.Conv1d(
            num_input_channels, num_affine_channels, kernel_size=1, bias=False
        )
        self.norm_1 = nn.BatchNorm1d(num_affine_channels)
        self.act_2 = utils.create_activation(activation_type)
        self.conv_3 = nn.Conv1d(
            num_affine_channels,
            num_input_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm_4 = nn.BatchNorm1d(num_input_channels, affine=False)

        self.a = nn.Conv1d(num_conditioning_channels, num_affine_channels, 1)
        self.b = nn.Conv1d(num_conditioning_channels, num_affine_channels, 1)

        nn.init.zeros_(self.a.weight)
        nn.init.ones_(self.a.bias)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

        self.random_normal_noise = None
        if with_random_noise:
            self.random_normal_noise = RandomNormalNoise(scale_size=noise_scale_size)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: A tensor of shape ``(N, C, T)``.
            conditioning: A tensor of shape ``(N, C)``.

        Returns:
            A tensor of shape ``(N, C, T)``.

        """
        residual = self.conv_0(x)
        residual = self.norm_1(residual)

        conditioning = conditioning[:, :, None]
        a = self.a(conditioning)
        b = self.b(conditioning)
        residual = a * residual + b

        if self.random_normal_noise is not None:
            residual = self.random_normal_noise(residual)

        residual = self.act_2(residual)
        residual = self.conv_3(residual)
        residual = self.norm_4(residual)

        return x + residual

    def enroll(self, conditioning: torch.Tensor):
        """

        Args:
            conditioning: A tensor of shape ``(N, C)``.

        Returns:

        """
        conditioning = conditioning[:, :, None]
        self.a = Buffer(self.a(conditioning))
        self.b = Buffer(self.b(conditioning))


class ConditionalResNet1d(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_conditioning_channels: int,
        num_affine_channels: int,
        num_residual_blocks: int,
        num_output_channels: int,
        upsampling_stride: Optional[int] = None,
        activation_type: str = "silu",
        with_random_noise: bool = False,
    ):
        super().__init__()

        self.conditional_modules = nn.ModuleList(
            modules=[
                ConditionalResidualBlock1d(
                    num_input_channels,
                    num_conditioning_channels,
                    num_affine_channels,
                    activation_type=activation_type,
                    with_random_noise=with_random_noise,
                )
                for _ in range(num_residual_blocks)
            ]
        )

        top = []
        if upsampling_stride is not None:
            top.append(
                nn.ConvTranspose1d(
                    num_input_channels,
                    num_input_channels,
                    upsampling_stride * 2,
                    stride=upsampling_stride,
                    padding=(upsampling_stride + 1) // 2,
                    output_padding=upsampling_stride % 2,
                )
            )
        top.extend(
            [
                nn.Conv1d(
                    num_input_channels, num_input_channels, kernel_size=3, padding=(1,)
                ),
                # We use LeakyReLU instead of activation_type by mistake in the demo.
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv1d(
                    num_input_channels, num_output_channels, kernel_size=3, padding=(1,)
                ),
            ]
        )
        self.top = nn.Sequential(*top)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: A tensor of shape ``(N, C, T)``.
            conditioning: A tensor of shape ``(N, C)``.

        Returns:
            A tensor of shape ``(N, C, T)``.

        """
        for conditional_module in self.conditional_modules:
            x = conditional_module(x, conditioning)

        x = self.top(x)

        return x

    def enroll(self, conditioning: torch.Tensor):
        """

        Args:
            conditioning: A tensor of shape ``(N, C)``.

        Returns:

        """
        for conditional_module in self.conditional_modules:
            conditional_module.enroll(conditioning)


def _crop_and_add(a, b):
    min_t = min(a.size(dim=-1), b.size(dim=-1))
    a = a[..., :min_t]
    b = b[..., :min_t]
    return a + b


class ConditionalVoiceSynthesisBackbone(nn.Module):
    def __init__(
        self,
        f0_floor: float,
        f0_ceiling: float,
        loudness_min: float,
        loudness_max: float,
        content_num_channels: int,
        num_output_channels: int,
        num_output_speaker_embedding_channels: int,
        f0_num_embeddings: int = 384,
        loudness_num_embeddings: int = 384,
        style_num_channels: int = 960,
        input_f0_key: str = dm.F0,
        input_loudness_key: str = dm.LOUDNESS,
        input_frame_content_key: str = dm.FRAME_CONTENT,
        input_global_style_key: str = dm.GLOBAL_STYLE,
        input_code_key: str = dm.CODE,
        input_speaker_embedding_key: str = dm.SPEAKER_EMBEDDING,
        output_code_key: str = dm.CODE,
    ):
        super().__init__()

        self._input_f0_key = input_f0_key
        self._input_loudness_key = input_loudness_key
        self._input_frame_content_key = input_frame_content_key
        self._input_global_style_key = input_global_style_key
        self._input_code_key = input_code_key
        self._input_speaker_embedding_key = input_speaker_embedding_key
        self._output_code_key = output_code_key

        self._f0_floor = f0_floor
        self._f0_ceiling = f0_ceiling
        self._log2_f0_floor = math.log2(self._f0_floor)
        self._log2_f0_ceiling = math.log2(self._f0_ceiling)
        self._log2_f0_quant_scale = (self._log2_f0_ceiling - self._log2_f0_floor) / (
            f0_num_embeddings - 3 - 1
        )
        self.f0_embedding = nn.Embedding(f0_num_embeddings, 256)

        self._loudness_min = loudness_min
        self._loudness_max = loudness_max
        self._loudness_quant_scale = (self._loudness_max - self._loudness_min) / (
            loudness_num_embeddings - 1
        )
        self.loudness_embedding = nn.Embedding(loudness_num_embeddings, 256)

        self.f0_and_loudness_bottom = nn.Sequential(
            nn.Conv1d(
                self.f0_embedding.embedding_dim + self.loudness_embedding.embedding_dim,
                256,
                kernel_size=3,
                padding=(1,),
            ),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=(1,)),
            nn.SiLU(inplace=True),
            ResidualBlock1d(256, dilation=1),
            ResidualBlock1d(256, dilation=3),
            ResidualBlock1d(256, dilation=1),
            ResidualBlock1d(256, dilation=3),
            ResidualBlock1d(256, dilation=1),
            ResidualBlock1d(256, dilation=3),
            ResidualBlock1d(256, dilation=1),
            ResidualBlock1d(256, dilation=3),
            nn.Conv1d(256, 256, kernel_size=3, padding=(1,)),
        )

        self.content_bottom = nn.Sequential(
            nn.Conv1d(content_num_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
        )

        self.dilated_blocks = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Conv1d(256, 256, kernel_size=3, padding=(1,), bias=False),
                    nn.InstanceNorm1d(256, affine=True),
                    ResidualBlock1d(256, dilation=1, normalization_type="in"),
                    ResidualBlock1d(256, dilation=3, normalization_type="in"),
                    ResidualBlock1d(256, dilation=9, normalization_type="in"),
                    ResidualBlock1d(256, dilation=1, normalization_type="in"),
                    ResidualBlock1d(256, dilation=3, normalization_type="in"),
                    ResidualBlock1d(256, dilation=9, normalization_type="in"),
                    nn.Conv1d(256, 256, kernel_size=3, padding=(1,), bias=False),
                    nn.InstanceNorm1d(256, affine=True),
                )
                for _ in range(2)
            ]
        )

        self.upsamplings = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
                )
                for _ in self.dilated_blocks
            ]
        )

        self.top = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=(1,)),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(256, 256, kernel_size=3, padding=(1,)),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, num_output_channels, kernel_size=3, padding=(1,)),
        )

        self.style_conditioned_top = ConditionalResNet1d(
            256,
            style_num_channels,
            768,
            6,
            num_output_channels,
            upsampling_stride=2,
            activation_type="silu",
            with_random_noise=True,
        )

        # For gradient-reversed global style prediction loss.
        self.global_style_top = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, style_num_channels),
        )

        # For gradient-reversed speaker embedding prediction loss.
        self.speaker_embedding_bottom = nn.Sequential(
            nn.Conv1d(256, 256, 4, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, 3, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, 4, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, 3, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, 4, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, 3, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
        )
        self.speaker_embedding_top = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_output_speaker_embedding_channels),
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        f0 = batch[self._input_f0_key]
        frame_content = batch[self._input_frame_content_key]
        loudness = batch[self._input_loudness_key]
        global_style = batch[self._input_global_style_key]
        code = batch.get(self._input_code_key)
        speaker_embedding = batch.get(self._input_speaker_embedding_key)

        frame_content_length = batch.get(
            dm.get_length_key(self._input_frame_content_key)
        )
        code_length = batch.get(dm.get_length_key(self._output_code_key))

        f0 = f0.squeeze(dim=1)
        loudness = loudness.squeeze(dim=1)

        f0_index = (
            (f0.log2() - self._log2_f0_floor) / self._log2_f0_quant_scale
        ).round().to(torch.int64) + 2
        f0_index.clamp_(1, self.f0_embedding.num_embeddings - 1)
        f0_index[f0 == 0] = 0
        f0_embedding = self.f0_embedding(f0_index).permute(0, 2, 1)

        loudness_embedding = self.loudness_embedding(
            ((loudness - self._loudness_min) / self._loudness_quant_scale)
            .round()
            .long()
            .clamp(0, self.loudness_embedding.num_embeddings - 1)
        ).permute(0, 2, 1)

        f0_and_loudness_bottom = self.f0_and_loudness_bottom(
            torch.cat([f0_embedding[..., ::2], loudness_embedding[..., ::2]], dim=1)
        )
        f0_and_loudness_bottoms = [
            F.avg_pool1d(F.pad(f0_and_loudness_bottom, [0, 3], mode="replicate"), 4, 4),
            F.avg_pool1d(F.pad(f0_and_loudness_bottom, [0, 1], mode="replicate"), 2, 2),
            f0_and_loudness_bottom,
        ]

        content_bottom = self.content_bottom(frame_content)

        net = _crop_and_add(content_bottom, f0_and_loudness_bottoms[0])
        net_low_resolution = net
        for dilated_block, upsampling, f0_and_loudness in zip(
            self.dilated_blocks,
            self.upsamplings,
            f0_and_loudness_bottoms[1:],
        ):
            net_low_resolution = net + dilated_block(net)
            net = upsampling(net_low_resolution)
            net = _crop_and_add(net, f0_and_loudness)

        top = self.top(net)
        style_conditioned_top = self.style_conditioned_top(net, global_style)

        output_code = (top + style_conditioned_top)[:, None]

        output_code_length_candidates = []
        if frame_content_length is not None:
            output_code_length_candidates.append(frame_content_length * 4)
        if code_length is not None:
            output_code_length_candidates.append(code_length)
        if output_code_length_candidates:
            output_code_length = torch.stack(output_code_length_candidates).amin(dim=0)
        else:
            output_code_length = None

        if code is not None:
            output_code = output_code[..., : code.size(dim=-1)]
            if output_code_length is not None:
                output_code_length = output_code_length.clamp_max(
                    output_code.size(dim=-1)
                )

        output_map = {self._output_code_key: output_code}
        if output_code_length is not None:
            output_map[dm.get_length_key(self._output_code_key)] = output_code_length

        reverse_gradient_global_style = self.global_style_top(
            grad_ops.reverse_gradient(net_low_resolution.mean(dim=-1))
        )
        loss_map = {
            "gradient_reversal/global_style_l1_loss": F.l1_loss(
                reverse_gradient_global_style, global_style
            )
        }

        if speaker_embedding is not None:
            speaker_embedding_net = self.speaker_embedding_bottom(
                grad_ops.reverse_gradient(net_low_resolution)
            )
            speaker_embedding_net = torch.cat(
                [speaker_embedding_net.mean(dim=-1), speaker_embedding_net.std(dim=-1)],
                dim=1,
            )
            reverse_gradient_speaker_embedding = self.speaker_embedding_top(
                speaker_embedding_net
            )
            loss_map[
                "gradient_reversal/speaker_embedding_negative_cosine_similarity_loss"
            ] = -F.cosine_similarity(
                reverse_gradient_speaker_embedding, speaker_embedding
            ).mean()

        output_map[dm.LOSS_MAP] = loss_map

        if self.training:
            output_map[dm.GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR] = {
                "code": (output_code, code)
            }
            if output_code_length is not None:
                output_map[
                    dm.get_length_key(dm.GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR)
                ] = output_code_length

        return output_map
