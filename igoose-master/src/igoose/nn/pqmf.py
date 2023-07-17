from typing import Optional

import numpy as np
from scipy import optimize
from scipy import signal
import torch
from torch import nn
from torch.nn import functional as F


def _create_prototype_window(kernel_size, cutoff, beta):
    # Use kaiser windows as PQMF prototype windows.
    # Reference: https://github.com/kan-bayashi/ParallelWaveGAN/blob/6a5f2f9e2f39421f385f88b9123260b8a46d87c0/parallel_wavegan/layers/pqmf.py#L15,
    return signal.firwin(kernel_size, cutoff, window=("kaiser", beta))


def _create_analysis_filter(num_subbands, kernel_size, cutoff, beta):
    prototype = _create_prototype_window(kernel_size, cutoff, beta)
    # See https://ccrma.stanford.edu/~jos/sasp/Pseudo_QMF_Cosine_Modulation_Filter.html.
    k = np.arange(num_subbands, dtype=float)[:, None]
    return (
        2
        * prototype
        * np.cos(
            (k + 0.5)
            * np.pi
            / num_subbands
            * (np.arange(kernel_size) - (kernel_size - 1) / 2)
            + ((-1) ** k) * np.pi / 4
        )
    )


class AnalysisFilterHyperparameterLoss:
    def __init__(self, num_subbands, kernel_size):
        self._num_subbands = num_subbands
        self._kernel_size = kernel_size

    def compute_residual(self, x):
        """Compute residuals for ``scipy.optimize.least_squares``.

        Args:
            x: A tuple of ``cutoff`` and ``beta`` for ``_create_prototype_window``.

        Returns:

        """
        cutoff, beta = x
        prototype = _create_prototype_window(self._kernel_size, cutoff, beta)
        # Reference: https://github.com/kan-bayashi/ParallelWaveGAN/issues/195#issuecomment-671408750.
        # Note that we check more points than the implementation in the above issue.
        prediction = np.convolve(prototype, prototype[::-1])[
            np.arange(
                self._kernel_size - 1,
                stop=2 * self._kernel_size - 1,
                step=2 * self._num_subbands,
            )
        ]
        target = np.zeros_like(prediction)
        target[0] = 0.5 / self._num_subbands
        # TODO(pwcheng): Is the objective valid for both odd and even kernel sizes?
        return prediction - target


class PQMF(nn.Module):
    """Pseudo-QMF Cosine Modulation Filter Bank."""

    def __init__(
        self,
        num_subbands: int,
        kernel_size: int,
        cutoff_and_beta_pair: Optional[tuple[float, float]] = None,
        trainable: bool = False,
        mode: str = "center",
    ):
        """

        Args:
            num_subbands:
            kernel_size:
            cutoff_and_beta_pair:
            trainable:
            mode: A string of the filtering mode:

                - ``'center'``: The before pad widths and the after pad widths of
                  convolutions for analysis and synthesis are approximately the same.
                - ``'causal_analysis'``: The convolution for analysis will be causal.
                  The convolution for synthesis will be non-causal.
                - ``'causal_synthesis'``: The convolution for analysis will be
                  non-causal. The convolution for synthesis will be causal.

        """
        super().__init__()
        self._num_subbands = num_subbands
        self._kernel_size = kernel_size
        self.stride = num_subbands

        # Note that we use larger pad widths to properly reconstruct heads and tails of
        # inputs.
        if mode == "center":
            self.conv_pad_left = (self._kernel_size - 1) // 2
            self.conv_pad_right = (self._kernel_size - 1) - self.conv_pad_left
        elif mode == "causal_analysis":
            self.conv_pad_left = self._kernel_size - 1
            self.conv_pad_right = 0
        elif mode == "causal_synthesis":
            self.conv_pad_left = 0
            self.conv_pad_right = self._kernel_size - 1
        else:
            raise ValueError(
                "``mode`` must be ``'center'``, ``'causal_analysis'`` or "
                "``'causal_synthesis'``."
            )

        # To properly reconstruct signals' boundaries, the synthesis stage's input
        # should be padded with at least
        # ``boundary_reconstruction_synthesis_input_pad_left`` and
        # ``boundary_reconstruction_synthesis_input_pad_right`` values on left and
        # right. Note that the output of ``forward`` will be properly padded if
        # ``reconstruct_boundary=True`` is set.
        self.boundary_reconstruction_synthesis_input_pad_left = (
            self.conv_pad_right // self.stride
        )
        self.boundary_reconstruction_synthesis_input_pad_right = (
            self.conv_pad_left // self.stride
        )
        # To properly reconstruct signals' boundaries, the analysis stage's input
        # should be padded with at least
        # ``boundary_reconstruction_analysis_input_pad_left`` and
        # ``boundary_reconstruction_analysis_input_pad_right`` values on left and
        # right.
        self.boundary_reconstruction_analysis_input_pad_left = (
            self.boundary_reconstruction_synthesis_input_pad_left * self.stride
        )
        self.boundary_reconstruction_analysis_input_pad_right = (
            self.boundary_reconstruction_synthesis_input_pad_right * self.stride
        )

        if cutoff_and_beta_pair:
            cutoff, beta = cutoff_and_beta_pair
        else:
            cutoff, beta = optimize.least_squares(
                AnalysisFilterHyperparameterLoss(
                    self._num_subbands, self._kernel_size
                ).compute_residual,
                np.array([0.5 / num_subbands, 5.0]),
                bounds=([np.nextafter(0.0, 1.0), 0.0], [np.nextafter(1.0, -1.0), 16.0]),
                ftol=np.finfo(float).eps,
                xtol=np.finfo(float).eps,
                gtol=np.finfo(float).eps,
            ).x

        analysis_filter = _create_analysis_filter(
            num_subbands, kernel_size, cutoff, beta
        )
        self.analysis_filter = nn.Parameter(
            torch.from_numpy(analysis_filter[:, None]).float()
        )
        self.synthesis_filter = nn.Parameter(
            self.analysis_filter.detach() * self._num_subbands
        )
        self.analysis_filter.requires_grad_(requires_grad=trainable)
        self.synthesis_filter.requires_grad_(requires_grad=trainable)

    def forward(
        self,
        x,
        reconstruct_boundary: bool = True,
        subband_slice: Optional[slice] = None,
    ):
        if reconstruct_boundary:
            pad = [
                (
                    self.conv_pad_left
                    + self.boundary_reconstruction_analysis_input_pad_left
                ),
                (
                    self.conv_pad_right
                    + self.boundary_reconstruction_analysis_input_pad_right
                ),
            ]
        else:
            pad = [self.conv_pad_left, self.conv_pad_right]

        analysis_filter = self.analysis_filter
        if subband_slice is not None:
            analysis_filter = analysis_filter[subband_slice]

        return F.conv1d(F.pad(x, pad), analysis_filter, stride=self.stride)

    def synthesize(
        self, x, length: Optional[int] = None, reconstruct_boundary: bool = True
    ):
        if reconstruct_boundary:
            analysis_pad_left = (
                self.conv_pad_left
                + self.boundary_reconstruction_analysis_input_pad_left
            )
            analysis_pad_right = (
                self.conv_pad_right
                + self.boundary_reconstruction_analysis_input_pad_right
            )
        else:
            analysis_pad_left = self.conv_pad_left
            analysis_pad_right = self.conv_pad_right

        y = F.conv_transpose1d(x, self.synthesis_filter, stride=self.stride)
        y = y[..., analysis_pad_left:]
        if length is not None:
            return y[..., :length]

        return y[..., : y.size(dim=-1) - (analysis_pad_right - (self.stride - 1))]
