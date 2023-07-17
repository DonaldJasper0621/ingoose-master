from typing import Any, Mapping

import torch
import torchmetrics

from igoose import data_map as dm


class L1Error(torchmetrics.MeanAbsoluteError):
    def __init__(self, prediction_key: str, target_data_key: str, **kwargs):
        super().__init__(**kwargs)

        self._prediction_key = prediction_key
        self._target_data_key = target_data_key

    def update(self, output_map: Mapping[str, Any], data_map: Mapping[str, Any]):
        prediction = output_map[self._prediction_key]
        target = data_map[self._target_data_key]

        target_length = data_map.get(dm.get_length_key(self._target_data_key))
        if target_length is None:
            super().update(prediction, target)
            return

        prediction_length = output_map.get(dm.get_length_key(self._prediction_key))
        if prediction_length is not None:
            assert torch.equal(prediction_length, target_length)

        for p, t, length in zip(prediction.split(1), target.split(1), target_length):
            super().update(p[..., :length], t[..., :length])
