from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from igoose import data_map as dm


class L1Loss(nn.Module):
    def __init__(self, prediction_key: str, target_key: str, loss_scale: float = 1.0):
        super().__init__()

        self._prediction_key = prediction_key
        self._target_key = target_key
        self._loss_scale = loss_scale

    def forward(
        self, output_map: Mapping[str, Any], data_map: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        prediction = output_map[self._prediction_key]
        target = data_map[self._target_key]

        prediction_length = output_map.get(dm.get_length_key(self._prediction_key))
        target_length = data_map.get(dm.get_length_key(self._target_key))
        length_candidates = []
        if prediction_length is not None:
            length_candidates.append(prediction_length)
        if target_length is not None:
            length_candidates.append(target_length)

        if length_candidates:
            l1_losses = F.l1_loss(prediction, target, reduction="none")

            length = torch.stack(length_candidates).amin(dim=0)
            mask = (
                torch.arange(prediction.size(dim=-1), device=prediction.device)
                < length[:, None]
            ).float()
            dims = l1_losses.size()[1:-1]
            normalization_term = length.sum() * dims.numel()
            l1_loss = (
                l1_losses * mask[(slice(None),) + (None,) * len(dims)]
            ).sum() / normalization_term.clamp_min(1.0)
        else:
            l1_loss = F.l1_loss(prediction, target)

        return {"l1_loss": l1_loss * self._loss_scale}
