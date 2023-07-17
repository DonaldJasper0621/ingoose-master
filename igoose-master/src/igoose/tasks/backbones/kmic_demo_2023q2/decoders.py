import pathlib
from typing import Any, Mapping

import torch
from torch import cuda
from torch import jit
from torch import nn

from igoose import data_map as dm


class PretrainedNSFResNetBackbone(nn.Module):
    def __init__(
        self,
        input_code_key: str = dm.CODE,
        input_f0_key: str = dm.F0,
        output_signal_key: str = dm.SIGNAL,
    ):
        super().__init__()

        self._input_code_key = input_code_key
        self._input_f0_key = input_f0_key
        self._output_signal_key = output_signal_key

        parent = pathlib.Path(__file__).parent
        self._pretrained_module_map = {
            "model_cpu": jit.load(parent / "pretrained_decoder_cpu.pt")
        }
        if cuda.is_available():
            self._pretrained_module_map["model_cuda"] = jit.load(
                parent / "pretrained_decoder_cuda.pt"
            )

        for module in self._pretrained_module_map.values():
            module.eval()

    @torch.no_grad()
    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # input_code: ``(N, ..., T)``.
        input_code = batch[self._input_code_key]
        # input_f0: ``(N, 1, T)``.
        input_f0 = batch[self._input_f0_key]

        if any(
            dm.get_length_key(key) in batch
            for key in [self._input_code_key, self._input_f0_key]
        ):
            raise NotImplementedError

        model = self._pretrained_module_map[f"model_{input_code.device.type}"]

        output_signal = model(input_code.flatten(start_dim=1, end_dim=-2), input_f0)

        return {self._output_signal_key: output_signal}
