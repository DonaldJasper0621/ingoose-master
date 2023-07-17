import pathlib
from typing import Any, Mapping, Optional, Sequence

from hydra import utils as hydra_utils
from pytorch_lightning import plugins
import pytorch_lightning as pl
import torch
from torch import nn

from igoose import data_map as dm
from igoose.tasks import utils
from igoose.trainer_plugins import precisions

CHECKPOINT_KEY_TASK_MODULE = "igoose/task_module"
CHECKPOINT_KEY_TASK_NAME = "igoose/task_name"


class TaskBase(pl.LightningModule):
    def __init__(
        self,
        optimizer_conf: Mapping[str, Any],
        scheduler_configuration_conf: Mapping[str, Any],
        loss_confs: Sequence[Mapping[str, Any]] = (),
        val_name_to_metric_conf_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        val_name_to_metric_conf_map = val_name_to_metric_conf_map or {}
        self._val_names = sorted(val_name_to_metric_conf_map)
        self.val_name_to_metric_collection = nn.ModuleDict()
        for val_name in self._val_names:
            metric_conf_map = val_name_to_metric_conf_map[val_name]
            self.val_name_to_metric_collection[val_name] = nn.ModuleDict(
                modules={
                    tag: hydra_utils.instantiate(metric_conf)
                    for tag, metric_conf in metric_conf_map.items()
                }
            )

        self.losses = nn.ModuleList(
            modules=list(map(hydra_utils.instantiate, loss_confs))
        )

    def configure_optimizers(self):
        (
            optimizer,
            scheduler_configuration,
        ) = utils.build_optimizer_and_scheduler_configuration(
            list(self.parameters()),
            self.hparams.optimizer_conf,
            self.hparams.scheduler_configuration_conf,
        )
        return [optimizer], [scheduler_configuration]

    def on_train_start(self):
        super().on_train_start()

        if self.trainer.global_step == 0:
            self.trainer.save_checkpoint(
                pathlib.Path(self.trainer.log_dir, "checkpoints", "initial.ckpt")
            )

    def training_step(self, batch, batch_idx):
        del batch_idx  # Unused.

        loss = self.compute_loss(self(batch), batch, log_scope="train/")

        self.log("train/loss", loss)

        return loss

    def training_step_end(self, step_output):
        super().training_step_end(step_output)

        precision_plugin = self.trainer.strategy.precision_plugin
        if isinstance(
            precision_plugin,
            precisions.MultipleGradScalersMixedPrecisionPlugin,
        ):
            for index, scaler in enumerate(precision_plugin.scalers):
                self.log(
                    f"train/grad_scaler_scale_{index}",
                    scaler.get_scale(),
                    prog_bar=True,
                )
        elif (
            isinstance(precision_plugin, plugins.MixedPrecisionPlugin)
            and precision_plugin.scaler is not None
        ):
            self.log(
                "train/grad_scaler_scale",
                precision_plugin.scaler.get_scale(),
                prog_bar=True,
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx  # Unused.

        output_map = self(batch)

        self._add_validation_summaries(output_map, batch)

        val_name = self._val_names[dataloader_idx]
        metric_collection = self.val_name_to_metric_collection[val_name]
        for metric in metric_collection.values():
            metric.update(output_map, batch)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        self._add_validation_metric_summaries()

    def on_save_checkpoint(self, checkpoint):
        model_metadata = {
            CHECKPOINT_KEY_TASK_MODULE: type(self).__module__,
            CHECKPOINT_KEY_TASK_NAME: type(self).__qualname__,
        }
        assert not model_metadata.keys() & checkpoint.keys()

        checkpoint.update(model_metadata)

    def _add_validation_summaries(self, output_map, batch):
        del self, output_map, batch  # Unused.

    def _add_validation_metric_summaries(self):
        for val_name, metric_collection in self.val_name_to_metric_collection.items():
            for tag, metric in metric_collection.items():
                result = metric.compute()
                metric.reset()

                if self.trainer.sanity_checking:
                    continue

                if isinstance(result, Mapping):
                    for key, value in result.items():
                        self.log(
                            f"validation/{val_name}/{tag}/{key}",
                            value,
                            prog_bar=False,
                            sync_dist=True,
                        )
                else:
                    self.log(
                        f"validation/{val_name}/{tag}",
                        result,
                        prog_bar=False,
                        sync_dist=True,
                    )

    def compute_loss(self, output, batch, log_scope=None):
        loss_map = {}
        for loss_module in self.losses:
            loss_map.update(loss_module(output, batch))

        output_loss_map = output.get(dm.LOSS_MAP, {})
        for key, output_loss in output_loss_map.items():
            loss_map[f"backbone_loss/{key}"] = output_loss.mean()

        if log_scope is not None:
            for key, loss in loss_map.items():
                self.log(f"{log_scope}/{key}", loss)

        if not loss_map:
            return torch.zeros((), device=self.device)

        return torch.stack(list(loss_map.values())).sum()
