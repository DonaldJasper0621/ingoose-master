import torch
from torch.utils import data

from igoose.utilities import my_itertools


class ConcatDataset(data.Dataset):
    def __init__(self, datasets, transforms=None):
        super().__init__()
        self._dataset = data.ConcatDataset(datasets)
        self._transforms = list(transforms or [])

    def __len__(self):
        return len(self._dataset)

    @torch.inference_mode()
    def __getitem__(self, index):
        data_map = self._dataset[index]

        for transform in self._transforms:
            data_map = transform(data_map)

        return data_map


class HeadDataset(data.Dataset):
    def __init__(self, dataset, stop: int, transforms=None):
        super().__init__()
        self._dataset = dataset
        self._stop = stop
        self._transforms = list(transforms or [])

    def __len__(self):
        return min(len(self._dataset), self._stop)

    @torch.inference_mode()
    def __getitem__(self, index):
        data_map = self._dataset[index]

        for transform in self._transforms:
            data_map = transform(data_map)

        return data_map


class InfiniteIterableDataset(data.IterableDataset):
    def __init__(self, dataset, maybe_random):
        super().__init__()
        self._dataset = dataset
        self._maybe_random = maybe_random

    @torch.inference_mode()
    def __iter__(self):
        return my_itertools.make_infinite_item_generator(
            self._dataset, self._maybe_random
        )


class RandomDatasetMixture(data.IterableDataset):
    def __init__(self, dataset_and_weight_pairs, transforms=None):
        super().__init__()
        self._dataset_to_weight = dict(dataset_and_weight_pairs)
        self._transforms = list(transforms or [])

    @torch.inference_mode()
    def __iter__(self):
        for data_map in my_itertools.make_multi_sources_maybe_random_item_generator(
            self._dataset_to_weight, weights=self._dataset_to_weight.values()
        ):
            for transform in self._transforms:
                data_map = transform(data_map)

            yield data_map
