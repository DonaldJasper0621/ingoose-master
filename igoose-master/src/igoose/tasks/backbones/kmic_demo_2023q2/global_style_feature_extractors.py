from typing import Any, Iterable, Mapping

import torch
from torch import nn

from igoose import data_map as dm
from igoose.tasks.backbones import utils


class MyInstanceNorm1d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        normalize_var: bool = True,
        affine: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.normalize_var = normalize_var
        self.affine = affine
        self.weight = None
        self.bias = None
        if self.affine:
            self.weight = nn.Parameter(torch.ones((self.num_channels,)))
            self.bias = nn.Parameter(torch.zeros((self.num_channels,)))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        x = x - mean

        std = None
        if self.normalize_var:
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            std = (var + self.eps).sqrt()
            x = x / std

        if self.affine:
            x = self.weight[:, None] * x + self.bias[:, None]

        return x, mean, std


class DilatedDownsamplingResidualBlock1d(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        kernel_size: int,
        dilations: Iterable[int],
        stride: int = 1,
        activation_type: str = "silu",
    ):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv1d(
                num_input_channels,
                num_output_channels,
                stride * 2,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm1d(num_output_channels),
        )

        for dilation in dilations:
            self.residual.append(utils.create_activation(activation_type))
            self.residual.append(
                nn.Conv1d(
                    num_output_channels,
                    num_output_channels,
                    kernel_size,
                    padding=kernel_size // 2 * dilation,
                    dilation=dilation,
                    bias=False,
                ),
            )
            self.residual.append(nn.BatchNorm1d(num_output_channels))

        self.shortcut = nn.Sequential(
            nn.Conv1d(
                num_input_channels,
                num_output_channels,
                stride * 2,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm1d(num_output_channels),
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class MultiScaleMeansNetwork(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 1,
        activation_type: str = "silu",
    ):
        super().__init__()

        self.bottom = nn.Sequential(
            nn.Conv1d(num_input_channels, 32, 8, stride=4),
            utils.create_activation(activation_type),
            nn.Conv1d(32, 32, 3),
            utils.create_activation(activation_type),
        )

        self.stages = nn.ModuleList(
            modules=[
                nn.Sequential(
                    DilatedDownsamplingResidualBlock1d(
                        32, 64, 3, (1, 3, 9), stride=2, activation_type=activation_type
                    ),
                ),
                nn.Sequential(
                    DilatedDownsamplingResidualBlock1d(
                        64, 64, 3, (1, 3, 9), stride=2, activation_type=activation_type
                    ),
                    DilatedDownsamplingResidualBlock1d(
                        64, 128, 3, (1, 3, 9), stride=2, activation_type=activation_type
                    ),
                ),
                nn.Sequential(
                    DilatedDownsamplingResidualBlock1d(
                        128,
                        128,
                        3,
                        (1, 3, 9),
                        stride=2,
                        activation_type=activation_type,
                    ),
                    DilatedDownsamplingResidualBlock1d(
                        128,
                        256,
                        3,
                        (1, 3, 9),
                        stride=2,
                        activation_type=activation_type,
                    ),
                ),
                nn.Sequential(
                    DilatedDownsamplingResidualBlock1d(
                        256,
                        256,
                        3,
                        (1, 3, 9),
                        stride=2,
                        activation_type=activation_type,
                    ),
                    DilatedDownsamplingResidualBlock1d(
                        256,
                        256,
                        3,
                        (1, 3, 9),
                        stride=2,
                        activation_type=activation_type,
                    ),
                ),
                nn.Sequential(
                    DilatedDownsamplingResidualBlock1d(
                        256,
                        256,
                        3,
                        (1, 3, 9),
                        stride=2,
                        activation_type=activation_type,
                    ),
                    DilatedDownsamplingResidualBlock1d(
                        256,
                        256,
                        3,
                        (1, 3, 9),
                        stride=2,
                        activation_type=activation_type,
                    ),
                ),
            ]
        )

        self.aggregations = nn.ModuleList(
            modules=[
                MyInstanceNorm1d(64, normalize_var=False, affine=True),
                MyInstanceNorm1d(128, normalize_var=False, affine=True),
                MyInstanceNorm1d(256, normalize_var=False, affine=True),
                MyInstanceNorm1d(256, normalize_var=False, affine=True),
                MyInstanceNorm1d(256, normalize_var=False, affine=True),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: A Tensor of shape ``(N, T)``.

        Returns:

        """
        x = self.bottom(x)

        conditioning = []
        for stage, aggregation in zip(self.stages, self.aggregations):
            x = stage(x)
            x, mean, _ = aggregation(x)
            conditioning.append(mean)

        return torch.cat(conditioning, dim=1).squeeze(dim=-1)


class MultiScaleMeans(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 1,
        activation_type: str = "silu",
        input_key: str = dm.SIGNAL,
        output_key: str = dm.GLOBAL_STYLE,
    ):
        super().__init__()

        self._input_key = input_key
        self._output_key = output_key

        self.network = MultiScaleMeansNetwork(
            num_input_channels=num_input_channels, activation_type=activation_type
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # x: ``(N, C, T)``.
        x = batch[self._input_key]
        x_length = batch.get(dm.get_length_key(self._input_key))

        if x_length is not None:
            # TODO(pwcheng):
            pass

        return {self._output_key: self.network(x)}
