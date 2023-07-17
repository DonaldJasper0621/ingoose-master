import torch


def reverse_gradient(x: torch.Tensor) -> torch.Tensor:
    x_detach = x.detach()
    return x_detach - x + x_detach
