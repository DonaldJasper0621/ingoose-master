import contextlib
from typing import Literal
from unittest import mock

from pytorch_lightning import plugins


class MultipleGradScalersMixedPrecisionPlugin(plugins.MixedPrecisionPlugin):
    def __init__(
        self, precision: Literal["16", 16, "bf16"], device: str, num_grad_scalers: int
    ):
        super().__init__(precision, device)

        self.scalers = [self.scaler] + [
            plugins.MixedPrecisionPlugin(precision, device).scaler
            for _ in range(1, num_grad_scalers)
        ]

    @contextlib.contextmanager
    def with_grad_scaler(self, grad_scaler_index: int):
        with mock.patch.object(self, "scaler", self.scalers[grad_scaler_index]):
            yield

    def state_dict(self):
        return {"scalers": [scaler.state_dict() for scaler in self.scalers]}

    def load_state_dict(self, state_dict):
        for scaler, scaler_state_dict in zip(self.scalers, state_dict["scalers"]):
            scaler.load_state_dict(scaler_state_dict)
