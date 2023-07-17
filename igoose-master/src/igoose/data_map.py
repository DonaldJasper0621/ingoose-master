from typing import Any, Mapping

import numpy as np
import torch
from torch import nested
from torch.utils.data._utils import collate

CODE = "code"
F0 = "f0"
FRAME_CONTENT = "frame_content"
GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR = "gan_name_to_fake_data_and_real_data_pair"
GLOBAL_STYLE = "global_style"
LOUDNESS = "loudness"
SAMPLE_RATE = "sample_rate"
SIGNAL = "signal"
SPEAKER_EMBEDDING = "speaker_embedding"
LOSS_MAP = "loss_map"
TARGET_CODE = "target_code"
TARGET_SIGNAL = "target_signal"


def get_length_key(key: str) -> str:
    return f"{key}/length"


def _collate_tensor_(
    key: str,
    tensors: list[torch.Tensor],
    output_batch_: dict[str, Any],
    pad_mode: str = "constant",
):
    if key in {
        CODE,
        F0,
        FRAME_CONTENT,
        LOUDNESS,
        SIGNAL,
        TARGET_CODE,
        TARGET_SIGNAL,
    }:
        time_dim = -1
    else:
        time_dim = None

    try:
        output_tensor = collate.collate_tensor_fn(tensors)
        output_length = None
    except RuntimeError as error:
        if time_dim is None:
            raise

        sizes = torch.as_tensor([t.size() for t in tensors])
        maybe_padded_dim_mask = torch.zeros((sizes.shape[1],), dtype=torch.bool)
        maybe_padded_dim_mask[time_dim] = True
        if not (sizes[0] == sizes).all(dim=0)[~maybe_padded_dim_mask].all():
            raise RuntimeError(
                f"Cannot collate {key} with element sizes ``{sizes.tolist()}``."
            ) from error

        if pad_mode == "constant":
            output_tensor = nested.as_nested_tensor(tensors).to_padded_tensor(0.0)
        else:
            padded_tensors = []
            for tensor, pad in zip(tensors, sizes.amax(dim=0) - sizes):
                padded_tensors.append(
                    torch.from_numpy(
                        np.pad(
                            tensor.numpy(),
                            torch.stack([torch.zeros_like(pad), pad], dim=1),
                            mode=pad_mode,
                        )
                    )
                )

            output_tensor = torch.stack(padded_tensors)

        output_length = sizes[:, time_dim].clone()

    output_batch_[key] = output_tensor

    if output_length is not None:
        output_batch_[get_length_key(key)] = output_length

    return output_batch_


def collate_fn(
    data_maps: list[Mapping[str, Any]], pad_mode: str = "constant"
) -> Mapping[str, Any]:
    list_batch = {}
    for key in data_maps[0]:
        list_batch[key] = [data_map[key] for data_map in data_maps]

    output_batch = {}
    for key, tensors in list_batch.items():
        _collate_tensor_(key, tensors, output_batch, pad_mode=pad_mode)

    return output_batch
