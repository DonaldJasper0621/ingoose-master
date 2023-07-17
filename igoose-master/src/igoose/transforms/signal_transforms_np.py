import collections
import copy
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from igoose import data_map as dm


class AtMostLengthNp:
    def __init__(
        self,
        length_max: int,
        random_crop: bool,
        signal_key_to_hop_length: Optional[Mapping[str, int]] = None,
        signal_key_to_output_padding: Optional[Mapping[str, int]] = None,
    ):
        self._length_max = length_max
        self._random_crop = random_crop
        self._signal_key_to_hop_length = dict(
            signal_key_to_hop_length or {dm.SIGNAL: 1}
        )
        self._hop_length_lcm = np.lcm.reduce(
            list(self._signal_key_to_hop_length.values())
        )
        if self._hop_length_lcm != max(self._signal_key_to_hop_length.values()):
            raise NotImplementedError

        self._signal_key_to_output_padding = dict(signal_key_to_output_padding or {})

        self._hop_length_to_keys = collections.defaultdict(list)
        for signal_key, hop_length in self._signal_key_to_hop_length.items():
            self._hop_length_to_keys[hop_length].append(signal_key)
        self._hop_length_to_keys = dict(self._hop_length_to_keys)
        if 1 not in self._hop_length_to_keys:
            raise NotImplementedError

    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        output_data_map = dict(data_map)

        length = data_map[self._hop_length_to_keys[1][0]].size(dim=-1)

        if length <= self._length_max:
            return output_data_map

        if self._random_crop:
            start = (
                torch.randint(
                    low=0,
                    high=(length - self._length_max) // self._hop_length_lcm + 1,
                    size=(),
                ).item()
                * self._hop_length_lcm
            )
        else:
            start = 0

        for key, hop_length in self._signal_key_to_hop_length.items():
            y_start = start // hop_length
            y_length = (
                self._length_max // hop_length
                + self._signal_key_to_output_padding.get(key, 0)
            )
            y_stop = y_start + y_length
            output_data_map[key] = data_map[key][..., y_start:y_stop]

        return output_data_map


class PadToAtLeastLengthNp:
    def __init__(
        self,
        length_min: int,
        signal_keys: Iterable[str] = (dm.SIGNAL,),
        **np_pad_kwargs,
    ):
        self._length_min = length_min
        self._signal_keys = sorted(set(signal_keys))
        self._np_pad_kwargs = copy.deepcopy(np_pad_kwargs)

    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        output_data_map = dict(data_map)

        for key in self._signal_keys:
            x = data_map[key]
            pad = max(0, self._length_min - x.shape[-1])
            if not pad:
                continue

            output_data_map[key] = np.pad(
                x, [[0, 0]] * (x.ndim - 1) + [[0, pad]], **self._np_pad_kwargs
            )

        return output_data_map
