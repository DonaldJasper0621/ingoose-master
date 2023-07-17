from torch import nn
from torchaudio import transforms


class KaiserBestResample(nn.Module):
    def __init__(self, orig_freq: int, new_freq: int):
        super().__init__()

        self.resample = transforms.Resample(
            orig_freq=orig_freq,
            new_freq=new_freq,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            beta=14.769656459379492,
        )

    def forward(self, x):
        return self.resample(x)
