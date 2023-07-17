import pathlib
from typing import Any, Mapping

import torch
from torch import cuda
from torch import jit
from torch import nn

from igoose import data_map as dm
from igoose.nn import resamplings


class PretrainedNeMoSTTConformerBackbone(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        input_signal_key: str = dm.SIGNAL,
        output_frame_content_key: str = dm.FRAME_CONTENT,
    ):
        super().__init__()

        self._input_signal_key = input_signal_key
        self._output_frame_content_key = output_frame_content_key

        parent = pathlib.Path(__file__).parent
        self._pretrained_module_map: dict[str, jit.ScriptModule] = {
            "model_cpu": jit.load(
                parent / "pretrained_frame_content_feature_extractor_cpu.pt",
            )
        }
        if cuda.is_available():
            self._pretrained_module_map["model_cuda"] = jit.load(
                parent / "pretrained_frame_content_feature_extractor_cuda.pt",
            )

        for module in self._pretrained_module_map.values():
            module.eval()

        self.resample = resamplings.KaiserBestResample(sample_rate, 22050)

    @torch.no_grad()
    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # input_signal: ``(N, 1, T)``.
        input_signal = batch[self._input_signal_key]
        input_signal_length = batch.get(dm.get_length_key(self._input_signal_key))

        n, c, t = input_signal.size()
        if c != 1:
            raise NotImplementedError

        if input_signal_length is None:
            input_signal_length = input_signal.new_full((n,), t, dtype=torch.int64)

        model = self._pretrained_module_map[f"model_{input_signal.device.type}"]

        frame_content, frame_content_length = model(
            self.resample(input_signal.squeeze(dim=1)), input_signal_length
        )

        return {
            self._output_frame_content_key: frame_content,
            dm.get_length_key(self._output_frame_content_key): frame_content_length,
        }
