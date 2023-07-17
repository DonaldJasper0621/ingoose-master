import collections
import concurrent.futures
import pathlib
import pickle

import numpy as np
from pytorch_lightning import utilities
import torch
from torch.utils import data
import tqdm

from igoose import data_map as dm
from igoose.ops import array_ops_np


def _maybe_as_tensor(x):
    if isinstance(x, np.ndarray):
        return array_ops_np.as_tensor(x)

    if isinstance(x, str):
        return x

    return torch.as_tensor(x)


_DATA_KEY_TO_AS_TENSOR_FUNCTION = collections.defaultdict(
    lambda: _maybe_as_tensor,
    {dm.SIGNAL: array_ops_np.as_audio_tensor},
)


class DatasetBase:
    def __init__(
        self,
        transforms_np=None,
        transforms=None,
        cache: bool = False,
        cache_folder=None,
        load_all_cache_files: bool = True,
        cache_len=None,
    ):
        """

        Args:
            transforms_np:
            transforms:
            cache:
            cache_folder:
            load_all_cache_files: If ``True`` all the cache files will be loaded in
                ``__init__``, else the cache files will be loaded on demand.
            cache_len: The length of the cache. Note that it can be different from the
                length of the dataset.
        """
        self._transforms_np = list(transforms_np or [])
        self._transforms = list(transforms or [])
        self._cache_folder = (
            pathlib.Path(cache_folder) if cache_folder is not None else None
        )
        self._load_all_cache_files = load_all_cache_files
        self._cache_len = cache_len

        self._cache = None
        self._cache_paths = None
        if cache:
            self._prepare_cache_()

    def _dump_cache_file(self, obj, path):
        del self  # Unused.

        with open(path, mode="wb") as file:
            return pickle.dump(obj, file)

    def _load_cache_file(self, path):
        del self  # Unused.

        with open(path, mode="rb") as file:
            return pickle.load(file)

    @utilities.rank_zero_only
    def _maybe_dump_cache_to_folder(self):
        if self._cache_folder is None:
            return

        with concurrent.futures.ThreadPoolExecutor() as executor:
            self._cache_folder.mkdir(parents=True)
            for _ in tqdm.tqdm(
                (
                    executor.map(
                        lambda index: self._dump_cache_file(
                            self._load_maybe_cached_data(index),
                            self._cache_paths[index],
                        ),
                        range(len(self._cache_paths)),
                    )
                    if self._cache is None
                    else executor.map(
                        self._dump_cache_file, self._cache, self._cache_paths
                    )
                ),
                desc=f"Dump {type(self).__name__} cache",
                total=len(self._cache_paths),
            ):
                pass

    def _prepare_cache_(self):
        cache_len = self._cache_len
        if cache_len is None:
            try:
                cache_len = len(self)
            except TypeError as type_error:
                raise ValueError("``cache_len`` should be given.") from type_error

        if self._cache_folder is not None:
            stem_length = len(str(cache_len - 1))
            self._cache_paths = [
                self._cache_folder / f"{index:0{stem_length}d}.pickle"
                for index in range(cache_len)
            ]
            if all(path.exists() for path in self._cache_paths):
                if self._load_all_cache_files:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        self._cache = list(
                            tqdm.tqdm(
                                executor.map(self._load_cache_file, self._cache_paths),
                                desc=f"Load {type(self).__name__} cache",
                                total=cache_len,
                            )
                        )
                return

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if self._cache_folder is None or self._load_all_cache_files:
                self._cache = list(
                    tqdm.tqdm(
                        executor.map(self._load_maybe_cached_data, range(cache_len)),
                        desc=f"Load {type(self).__name__} cache",
                        total=cache_len,
                    )
                )

        self._maybe_dump_cache_to_folder()

    def _to_data_map(self, data_map_np):
        del self  # Unused.

        return {
            key: _DATA_KEY_TO_AS_TENSOR_FUNCTION[key](value_np)
            for key, value_np in data_map_np.items()
        }

    def _get_maybe_transformed_data_map(self, data_map_np):
        for transform_np in self._transforms_np:
            data_map_np = transform_np(data_map_np)

        data_map = self._to_data_map(data_map_np)

        for transform in self._transforms:
            data_map = transform(data_map)

        return data_map

    def _load_maybe_cached_data(self, cached_data_index):
        raise NotImplementedError

    def get_maybe_cached_data(self, cached_data_index):
        if self._cache is not None:
            return self._cache[cached_data_index]

        if self._cache_paths is not None:
            return self._load_cache_file(self._cache_paths[cached_data_index])

        return self._load_maybe_cached_data(cached_data_index)


class Dataset(DatasetBase, data.Dataset):
    def get_data_map_np(self, item_index):
        raise NotImplementedError

    @torch.inference_mode()
    def __getitem__(self, index):
        data_map_np = self.get_data_map_np(index)
        return self._get_maybe_transformed_data_map(data_map_np)


class IterableDataset(DatasetBase, data.IterableDataset):
    def make_data_map_np_generator(self):
        raise NotImplementedError

    def make_data_map_generator(self):
        yield from map(
            self._get_maybe_transformed_data_map, self.make_data_map_np_generator()
        )

    @torch.inference_mode()
    def __iter__(self):
        yield from self.make_data_map_generator()
