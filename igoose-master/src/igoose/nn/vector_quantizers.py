from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import utils


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        code_num_channels: int,
        num_codes: int,
        with_weight_norm: bool = True,
    ):
        super().__init__()

        self.input_linear = nn.Conv1d(num_input_channels, code_num_channels, 1)
        self.output_linear = nn.Conv1d(code_num_channels, num_input_channels, 1)

        self.codebook = nn.Embedding(num_codes, code_num_channels)

        if with_weight_norm:
            self.input_linear = utils.weight_norm(self.input_linear)
            self.output_linear = utils.weight_norm(self.output_linear)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            x: A tensor of shape ``(N, C, ...)``.

        Returns:

        """
        n, _, *dims = x.size()
        x_3d = x[..., None].flatten(start_dim=2)

        z_e = self.input_linear(x_3d)

        z_q, code_index = self.embed_continuous_features(z_e)

        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean(dim=(1, 2))
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean(
            dim=(1, 2)
        )

        if self.training:
            z_q_forward = z_q
            z_q_backward = z_e
            z_q = z_q_forward.detach() + (z_q_backward - z_q_backward.detach())

        y_3d = self.output_linear(z_q)

        y = y_3d.view_as(x)
        code_index = code_index.view(n, *dims)

        return y, code_index, codebook_loss, commitment_loss

    def embed_code(self, code_index: torch.Tensor) -> torch.Tensor:
        return self.codebook(code_index).movedim(-1, 1)

    def embed_continuous_features(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            z_e: A tensor of shape ``(N, C, ...)``.

        Returns:
            A tensor of the quantized features and a tensor of the code indices.
        """
        code_index = F.cosine_similarity(
            z_e.movedim(1, -1)[..., :, None, :], self.codebook.weight, dim=-1
        ).argmax(dim=-1)
        return self.embed_code(code_index), code_index


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        codebooks_code_num_channels_and_num_codes_pairs: Iterable[tuple[int, int]],
        quantizer_dropout_probability: float = 0.0,
        random_num_quantizers_min: int = 1,
    ):
        """

        Args:
            num_input_channels:
            codebooks_code_num_channels_and_num_codes_pairs:
            quantizer_dropout_probability: See https://arxiv.org/abs/2107.03312.
            random_num_quantizers_min:
        """
        super().__init__()

        self._quantizer_dropout_probability = quantizer_dropout_probability
        self._random_num_quantizers_min = random_num_quantizers_min

        self.vqs = nn.ModuleList(
            modules=[
                VectorQuantizer(num_input_channels, code_num_channels, num_codes)
                for code_num_channels, num_codes in (
                    codebooks_code_num_channels_and_num_codes_pairs
                )
            ]
        )

    def forward(
        self, x, num_quantizers: Optional[int] = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """

        Args:
            x: A tensor of shape ``(N, C, ...)``.
            num_quantizers: See https://arxiv.org/abs/2107.03312.

        Returns:

        """
        quantizers = self.vqs
        if num_quantizers is not None:
            quantizers = quantizers[:num_quantizers]

        device = x.device
        n = x.size(dim=0)
        if self.training:
            random_num_quantizers = torch.where(
                torch.rand((n,), device=device) < self._quantizer_dropout_probability,
                torch.randint(
                    self._random_num_quantizers_min,
                    len(self.vqs) + 1,
                    (n,),
                    device=device,
                ),
                len(self.vqs),
            )
            quantizer_mask = (
                torch.arange(len(quantizers), device=device)[:, None]
                < random_num_quantizers
            )
        else:
            quantizer_mask = torch.ones(
                (len(quantizers), n), dtype=torch.bool, device=device
            )

        z_q = torch.zeros((), device=device)
        residual = x
        code_indices = []
        codebook_losses = []
        commitment_losses = []
        for vq, mask in zip(self.vqs, quantizer_mask):
            r_z_q, r_code_index, r_codebook_loss, r_commitment_loss = vq(residual)

            code_indices.append(r_code_index)
            codebook_losses.append(r_codebook_loss)
            commitment_losses.append(r_commitment_loss)

            z_q = z_q + r_z_q * mask.view(n, *(1,) * (x.ndim - 1))

            residual = residual - r_z_q

        codebook_loss = (
            (quantizer_mask * torch.stack(codebook_losses)).sum(dim=0).mean()
        )
        commitment_loss = (
            (quantizer_mask * torch.stack(commitment_losses)).sum(dim=0).mean()
        )

        return z_q, code_indices, codebook_loss, commitment_loss
