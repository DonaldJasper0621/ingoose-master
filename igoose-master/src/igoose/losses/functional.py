import torch
from torch.nn import functional as F


def lsgan_as_fake_data_loss(
    discriminator_output: torch.Tensor, reduction: str = "mean"
):
    return F.mse_loss(
        discriminator_output,
        discriminator_output.new_tensor(0.0).expand_as(discriminator_output),
        reduction=reduction,
    )


def lsgan_as_real_data_loss(
    discriminator_output: torch.Tensor, reduction: str = "mean"
):
    return F.mse_loss(
        discriminator_output,
        discriminator_output.new_tensor(1.0).expand_as(discriminator_output),
        reduction=reduction,
    )
