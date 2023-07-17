import copy
import functools
from typing import Any, Mapping, Sequence

from hydra import utils
import pytorch_lightning as pl
from torch.utils import data

from igoose import data_map as dm


def _create_data_loader(dataset, pad_mode="constant", **data_loader_kwargs):
    data_loader_kwargs.setdefault(
        "collate_fn", functools.partial(dm.collate_fn, pad_mode=pad_mode)
    )
    return data.DataLoader(dataset, **data_loader_kwargs)


def _has_same_keys(*mappings):
    return len({frozenset(mapping.keys()) for mapping in mappings}) == 1


class BasicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_conf: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        train_data_loader_kwargs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        val_name_to_dataset_conf: Mapping[str, Any],
        val_name_to_data_loader_kwargs: Mapping[str, Any],
        val_name_to_metric_conf_map: Mapping[str, Mapping[str, Any]],
        collate_fn_pad_mode: str = "constant",
    ):
        super().__init__()
        self._train_dataset_conf = copy.deepcopy(train_dataset_conf)
        self._train_data_loader_kwargs = copy.deepcopy(train_data_loader_kwargs)

        if not _has_same_keys(
            val_name_to_dataset_conf,
            val_name_to_data_loader_kwargs,
            val_name_to_metric_conf_map,
        ):
            raise ValueError(
                "Names of ``val_name_to_dataset_conf``, "
                "``val_name_to_data_loader_kwargs`` and ``val_name_to_metric_conf_map``"
                " should be the same."
            )

        self._val_dataset_confs = []
        self._val_data_loaders_kwargs = []
        for name in sorted(val_name_to_dataset_conf):
            self._val_dataset_confs.append(
                copy.deepcopy(val_name_to_dataset_conf[name])
            )
            self._val_data_loaders_kwargs.append(
                copy.deepcopy(val_name_to_data_loader_kwargs[name])
            )

        self._collate_fn_pad_mode = collate_fn_pad_mode

        self._train_dataset = None
        self._train_datasets = None
        self._val_datasets = None

    def setup(self, stage=None):
        if isinstance(self._train_dataset_conf, Sequence):
            self._train_datasets = utils.instantiate(self._train_dataset_conf)
        else:
            self._train_dataset = utils.instantiate(self._train_dataset_conf)
        self._val_datasets = list(map(utils.instantiate, self._val_dataset_confs))

    def train_dataloader(self, *args, **kwargs):
        del args, kwargs  # Unused.

        if self._train_dataset is not None:
            return _create_data_loader(
                self._train_dataset,
                pad_mode=self._collate_fn_pad_mode,
                **self._train_data_loader_kwargs,
            )

        return [
            _create_data_loader(
                train_dataset,
                pad_mode=self._collate_fn_pad_mode,
                **train_data_loader_kwargs,
            )
            for train_dataset, train_data_loader_kwargs in zip(
                self._train_datasets, self._train_data_loader_kwargs
            )
        ]

    def val_dataloader(self, *args, **kwargs):
        del args, kwargs  # Unused.

        return [
            _create_data_loader(
                dataset, pad_mode=self._collate_fn_pad_mode, **data_loader_kwargs
            )
            for dataset, data_loader_kwargs in zip(
                self._val_datasets, self._val_data_loaders_kwargs
            )
        ]
