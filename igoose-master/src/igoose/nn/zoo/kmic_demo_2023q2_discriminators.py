import torch
from torch import jit
from torch import nn
from torch.nn import utils


class TimeStride16CNN2dDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            utils.weight_norm(
                nn.Conv2d(1, 32, (7, 4), stride=(1, 2), padding=(3, 1)),
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            utils.weight_norm(
                nn.Conv2d(32, 32, (8, 4), stride=(2, 2), padding=(3, 1)),
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            utils.weight_norm(
                nn.Conv2d(32, 64, (8, 4), stride=(2, 2), padding=(3, 1)),
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            utils.weight_norm(
                nn.Conv2d(64, 64, (8, 4), stride=(2, 2), padding=(3, 1)),
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            utils.weight_norm(nn.Conv2d(64, 64, (3, 3), padding=(1, 1))),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            utils.weight_norm(nn.Conv2d(64, 1, (3, 3), padding=(1, 1), bias=True)),
        )

    def forward(self, x):
        return self.sequential(x)

    @jit.export
    def compute_output_length(
        self, input_length: int | torch.Tensor
    ) -> int | torch.Tensor:
        return input_length // 16
