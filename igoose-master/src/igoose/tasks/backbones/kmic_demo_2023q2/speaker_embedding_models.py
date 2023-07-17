import pathlib
from typing import Any, Mapping

import torch
from torch import cuda
from torch import jit
from torch import nn

from igoose import data_map as dm
from igoose.nn import resamplings


class PretrainedSpeakerEmbeddingBackbone(nn.Module):
    """

    Do not use the model in products.

    """

    def __init__(
        self,
        sample_rate: int,
        input_signal_key: str = dm.SIGNAL,
        output_embedding_key: str = dm.SPEAKER_EMBEDDING,
    ):
        super().__init__()

        self._input_signal_key = input_signal_key
        self._output_embedding_key = output_embedding_key

        parent = pathlib.Path(__file__).parent
        model_path = parent / "pretrained_speaker_embedding_model.pt"
        self._pretrained_module_map = {"model_cpu": jit.load(model_path)}
        if cuda.is_available():
            self._pretrained_module_map["model_cuda"] = jit.load(model_path).cuda()

        for module in self._pretrained_module_map.values():
            module.eval()

        self.resample = resamplings.KaiserBestResample(sample_rate, 16000)

    @property
    def num_channels(self):
        return 5120

    @torch.no_grad()
    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # input_signal: ``(N, 1, T)``.
        input_signal = batch[self._input_signal_key]

        if dm.get_length_key(input_signal) in batch:
            raise NotImplementedError

        model = self._pretrained_module_map[f"model_{input_signal.device.type}"]

        _, pool_mean, pool_std = model(self.resample(input_signal.squeeze(dim=1)))
        output_embedding = torch.cat([pool_mean, pool_std], dim=1).flatten(start_dim=1)

        return {self._output_embedding_key: output_embedding}
