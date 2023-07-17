from typing import Optional

from torch import nn


def create_activation(
    activation_type: str, num_channels: Optional[int] = None, maybe_inplace: bool = True
):
    del num_channels  # Unused.

    match activation_type:
        case "leaky_relu":
            return nn.LeakyReLU(inplace=maybe_inplace)

        case "relu":
            return nn.ReLU(inplace=maybe_inplace)

        case "silu":
            return nn.SiLU(inplace=maybe_inplace)

        case _:
            raise NotImplementedError


def create_normalization(
    activation_type: str,
    num_channels: int,
    affine: bool = True,
    num_dimensions: int = 1,
):
    normalization_cls = {
        ("bn", 1): nn.BatchNorm1d,
        ("bn", 2): nn.BatchNorm2d,
        ("in", 1): nn.InstanceNorm1d,
        ("in", 2): nn.InstanceNorm2d,
    }[activation_type, num_dimensions]

    return normalization_cls(num_channels, affine=affine)
