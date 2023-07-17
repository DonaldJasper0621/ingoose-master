from typing import Optional

from scipy import fft
import torch
from torch import jit
from torch.nn import functional as F
from torchaudio import functional


@jit.ignore
def get_fft_next_fast_len(target: int, real: Optional[bool] = False) -> int:
    return fft.next_fast_len(target, real)


def fft_convolve(x: torch.Tensor, y: torch.Tensor):
    x_length = x.size(dim=-1)
    y_length = y.size(dim=-1)
    output_length = x_length + y_length - 1
    fft_length = get_fft_next_fast_len(output_length, True)
    fx = torch.fft.rfft(F.pad(x, [0, fft_length - x_length]), dim=-1)
    fy = torch.fft.rfft(F.pad(y, [0, fft_length - y_length]), dim=-1)
    return torch.fft.irfft(fx * fy, n=fft_length)[..., :output_length]


def safe_normalize_magnitude(
    signal: torch.Tensor, magnitude: float | torch.Tensor, dim=-1
):
    """

    Args:
        signal:
        magnitude: A float or a tensor broadcastable to ``signal``.
        dim:

    Returns:

    """
    input_magnitude_max = signal.abs().amax(dim=dim, keepdim=True)
    scale = (magnitude / input_magnitude_max).nan_to_num(
        nan=0.0, posinf=0.0, neginf=0.0
    )
    return signal * scale, scale


def resample_kaiser_best(
    x: torch.Tensor, input_sample_rate: int, output_sample_rate: int
) -> torch.Tensor:
    """See ``librosa.resample`` with ``res_type=kaiser_best``.

    Args:
        x:
        input_sample_rate:
        output_sample_rate:

    Returns:

    """
    return functional.resample(
        x,
        orig_freq=input_sample_rate,
        new_freq=output_sample_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


def resample_rfft(x: torch.Tensor, num_output_samples: int, dim: int = -1):
    """Apply ``scipy.signal.resample`` on real-valued inputs.

    Args:
        x:
        num_output_samples:
        dim:

    Returns:

    """
    x_f = torch.fft.rfft(x, dim=dim)

    y_f_size = list(x.size())
    y_f_size[dim] = num_output_samples // 2 + 1
    y_f = x_f.new_zeros(y_f_size)

    num_input_samples = x.size(dim=dim)
    n = min(num_input_samples, num_output_samples)
    y_f.narrow(dim, 0, n // 2 + 1)[:] = x_f.narrow(dim, 0, n // 2 + 1)

    if n % 2 == 0:
        if num_output_samples < num_input_samples:
            y_f.narrow(dim, n // 2, 1)[:] *= 2.0
        elif num_input_samples < num_output_samples:
            y_f.narrow(dim, n // 2, 1)[:] *= 0.5

    y = torch.fft.irfft(y_f, n=num_output_samples, dim=dim)
    y *= num_output_samples / num_input_samples
    return y


def linear_upsample(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    x_3d = x[None, None, None].flatten(end_dim=-3)
    padded_x_3d = F.pad(x_3d, [0, 1], mode="replicate")
    padded_output_3d = F.interpolate(
        padded_x_3d,
        size=x_3d.size(dim=-1) * scale_factor + 1,
        mode="linear",
        align_corners=True,
    )
    return padded_output_3d[..., :-1].view(*x.size()[:-1], -1)
