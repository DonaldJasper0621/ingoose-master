import torch
from torch import jit
from torch import nn
from torch.nn import utils
from torchaudio import transforms


class MPDDiscriminator(nn.Module):
    """See https://arxiv.org/abs/2010.05646."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 6,
        stride: int = 3,
        with_spectral_norm: bool = False,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()

        self._period = period

        kernel_size_2d = (1, kernel_size)
        stride_2d = (1, stride)
        padding_2d = (0, (kernel_size - 1) // 2)
        norm = utils.spectral_norm if with_spectral_norm else utils.weight_norm
        self.sequential = nn.Sequential(
            norm(nn.Conv2d(1, 32, kernel_size_2d, stride_2d, padding=padding_2d)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(nn.Conv2d(32, 128, kernel_size_2d, stride_2d, padding=padding_2d)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(nn.Conv2d(128, 512, kernel_size_2d, stride_2d, padding=padding_2d)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(nn.Conv2d(512, 1024, kernel_size_2d, stride_2d, padding=padding_2d)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(nn.Conv2d(1024, 1024, (1, 5), 1, padding=(0, 2))),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(nn.Conv2d(1024, 1, (1, 3), 1, padding=(0, 1))),
        )

    def forward(self, x):
        b, c, t = x.size()
        p = self._period
        x = x[:, :, : t - t % p].view(b, c, t // p, p)
        # Move time axis to the last dimension.
        x = x.permute(0, 1, 3, 2)
        return self.sequential(x)

    @jit.export
    def compute_output_length(
        self, input_length: int | torch.Tensor
    ) -> int | torch.Tensor:
        return input_length // (self._period * 3 * 3 * 3 * 3)


class MRDDiscriminator(nn.Module):
    """See https://arxiv.org/abs/2106.07889."""

    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        with_spectral_norm: bool = False,
        discriminator_channel_mult: int = 1,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()

        self.stft = transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,
            center=True,
        )

        norm = utils.spectral_norm if with_spectral_norm else utils.weight_norm
        num_channels = int(32 * discriminator_channel_mult)
        self.sequential = nn.Sequential(
            norm(nn.Conv2d(1, num_channels, (3, 9), padding=(1, 4))),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(
                nn.Conv2d(
                    num_channels, num_channels, (3, 10), stride=(1, 2), padding=(1, 4)
                )
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(
                nn.Conv2d(
                    num_channels, num_channels, (3, 10), stride=(1, 2), padding=(1, 4)
                )
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(
                nn.Conv2d(
                    num_channels, num_channels, (3, 10), stride=(1, 2), padding=(1, 4)
                )
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1))),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            norm(nn.Conv2d(num_channels, 1, (3, 3), padding=(1, 1))),
        )

    def forward(self, x):
        return self.sequential(self.stft(x).abs())

    @jit.export
    def compute_output_length(
        self, input_length: int | torch.Tensor
    ) -> int | torch.Tensor:
        stft = self.stft
        spectrogram_length = (input_length - stft.n_fft % 1) // stft.hop_length + 1
        return spectrogram_length // 8
