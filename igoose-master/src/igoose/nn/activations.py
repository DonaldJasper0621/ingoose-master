import torch
from torch import nn
from torch.autograd import function


class _ClampFunction(function.Function):
    @staticmethod
    def forward(ctx, x, value_min, value_max):
        ctx.save_for_backward(x, value_min, value_max)
        return x.clamp(min=value_min, max=value_max)

    @staticmethod
    def backward(ctx, grad_output):
        x, value_min, value_max = ctx.saved_tensors

        # We assume ``grad_output`` is a gradient of a loss function being minimized.
        mask = torch.zeros_like(x, dtype=torch.bool)
        if value_min is not None:
            mask |= (x < value_min) & (grad_output > 0.0)
        if value_max is not None:
            mask |= (x > value_max) & (grad_output < 0.0)
        grad_input = grad_output.clone()
        grad_input[mask] = 0.0

        return grad_input, None, None


class Clamp(nn.Module):
    def __init__(
        self, value_min=None, value_max=None, maybe_keep_clamped_value_gradient=False
    ):
        """

        Args:
            value_min:
            value_max:
            maybe_keep_clamped_value_gradient: A boolean indicating if we keep
                positive gradients of values greater than ``value_max`` and negative
                gradients of values less than ``value_min``.
        """
        if value_min is None and value_max is None:
            raise ValueError(
                "At least one of `value_min` or `value_max` must not be `None`."
            )

        super().__init__()

        self._maybe_keep_clamped_value_gradient = maybe_keep_clamped_value_gradient

        if value_min is not None:
            value_min = torch.as_tensor(value_min).clone()
        if value_max is not None:
            value_max = torch.as_tensor(value_max).clone()
        self.register_buffer("value_min", value_min)
        self.register_buffer("value_max", value_max)

    def forward(self, x):
        if self.training and self._maybe_keep_clamped_value_gradient:
            return _ClampFunction.apply(x, self.value_min, self.value_max)

        return torch.clamp(x, min=self.value_min, max=self.value_max)

    def extra_repr(self) -> str:
        return (
            f"min={self.value_min},max={self.value_max},"
            "maybe_keep_clamped_value_gradient="
            f"{self._maybe_keep_clamped_value_gradient}"
        )
