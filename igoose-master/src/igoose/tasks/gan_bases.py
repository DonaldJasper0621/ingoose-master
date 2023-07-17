import contextlib
import dataclasses
import itertools
from typing import Any, Mapping, Optional, Sequence

from hydra import utils as hydra_utils
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchvision.models import feature_extraction

from igoose import data_map as dm
from igoose.losses import functional
from igoose.tasks import bases
from igoose.tasks import utils
from igoose.trainer_plugins import precisions


@dataclasses.dataclass
class GANSpec:
    discriminator_conf: Mapping[str, Any]
    discriminator_gan_loss_scale: float
    generator_gan_loss_scale: float
    fake_data_and_real_data_pair_names: Sequence[str]
    feature_matching_loss_scale: float = 0.0
    feature_matching_node_names: Sequence[str] = ()
    feature_matching_node_name_to_time_stride: Optional[Mapping[str, int]] = None
    feature_matching_node_name_to_loss_scale: Optional[Mapping[str, float]] = None
    score_node_name: Optional[str] = None

    def __post_init__(self):
        if self.feature_matching_loss_scale and not self.feature_matching_node_names:
            raise ValueError("Empty ``feature_matching_node_names``.")

        if self.feature_matching_loss_scale and self.score_node_name is None:
            raise ValueError(
                "``score_node_name`` must be given when feature matching losses are "
                "enabled as we will use "
                "``torchvision.models.feature_extraction.create_feature_extractor`` to "
                "create a discriminator graph module returning intermediate features "
                "and GAN scores."
            )

        feature_matching_node_name_to_loss_scale = dict(
            self.feature_matching_node_name_to_loss_scale or {}
        )
        self.feature_matching_node_name_to_loss_scale = {
            name: (
                self.feature_matching_loss_scale
                * feature_matching_node_name_to_loss_scale.get(name, 1.0)
            )
            for name in self.feature_matching_node_names
        }


def _compute_time_mask(score: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    t = score.size(dim=-1)
    mask = torch.arange(t, device=length.device) < length[:, None]
    return mask.view(-1, *(1,) * (score.ndim - 2), t).float()


class _GANHead(nn.Module):
    def __init__(self, spec: GANSpec):
        super().__init__()

        self._spec = spec
        self.discriminator = hydra_utils.instantiate(self._spec.discriminator_conf)
        self._with_feature_matching_loss = self._spec.feature_matching_loss_scale != 0.0
        self.discriminator_feature_extractor = None
        if self._with_feature_matching_loss:
            self.discriminator_feature_extractor = (
                feature_extraction.create_feature_extractor(
                    self.discriminator,
                    return_nodes=list(
                        {*spec.feature_matching_node_names, spec.score_node_name}
                    ),
                )
            )
            assert set(self.discriminator_feature_extractor.parameters()) <= set(
                self.discriminator.parameters()
            )

    def maybe_compute_discriminator_loss(self, output, batch):
        del batch  # Unused.

        loss_map = {}
        gan_loss_scale = self._spec.discriminator_gan_loss_scale
        data_pair_map = output[dm.GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR]
        length = output.get(
            dm.get_length_key(dm.GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR)
        )
        for name in self._spec.fake_data_and_real_data_pair_names:
            try:
                fake_data, real_data = data_pair_map[name]
            except KeyError:
                if not self.training:
                    continue

                raise

            fake_data = fake_data.detach()
            real_data = real_data.detach()

            scores = self.discriminator(torch.cat([fake_data, real_data]))
            fake_score, real_score = scores.chunk(2)

            if length is None:
                as_fake_data_loss = gan_loss_scale * functional.lsgan_as_fake_data_loss(
                    fake_score
                )
                as_real_data_loss = gan_loss_scale * functional.lsgan_as_real_data_loss(
                    real_score
                )
            else:
                score_length = self.discriminator.compute_output_length(length)
                score_mask = _compute_time_mask(fake_score, score_length)
                scale = gan_loss_scale / (
                    fake_score.size()[1:-1].numel() * score_length.sum()
                ).clamp_min(1.0)

                as_fake_data_losses = functional.lsgan_as_fake_data_loss(
                    fake_score, reduction="none"
                )
                as_fake_data_loss = scale * (as_fake_data_losses * score_mask).sum()
                as_real_data_losses = functional.lsgan_as_real_data_loss(
                    real_score, reduction="none"
                )
                as_real_data_loss = scale * (as_real_data_losses * score_mask).sum()

            loss_map.update(
                {
                    f"D/GAN/as_fake_data_loss/{name}": as_fake_data_loss,
                    f"D/GAN/as_real_data_loss/{name}": as_real_data_loss,
                }
            )

        return loss_map

    def maybe_compute_generator_loss(self, output, batch):
        del batch  # Unused.

        loss_map = {}
        gan_loss_scale = self._spec.generator_gan_loss_scale
        node_name_to_feature_matching_loss_scale = (
            self._spec.feature_matching_node_name_to_loss_scale
        )
        data_pair_map = output[dm.GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR]
        length = output.get(
            dm.get_length_key(dm.GAN_NAME_TO_FAKE_DATA_AND_REAL_DATA_PAIR)
        )
        for name in self._spec.fake_data_and_real_data_pair_names:
            try:
                fake_data, real_data = data_pair_map[name]
            except KeyError:
                if not self.training:
                    continue

                raise

            if self._with_feature_matching_loss:
                node_name_to_fake_feature = self.discriminator_feature_extractor(
                    fake_data
                )
                fake_score = node_name_to_fake_feature[self._spec.score_node_name]

                with torch.no_grad():
                    node_name_to_real_feature = self.discriminator_feature_extractor(
                        real_data
                    )
            else:
                node_name_to_fake_feature = {}
                fake_score = self.discriminator(fake_data)
                node_name_to_real_feature = {}

            if length is None:
                as_real_data_loss = gan_loss_scale * functional.lsgan_as_real_data_loss(
                    fake_score
                )
            else:
                score_length = self.discriminator.compute_output_length(length)
                score_mask = _compute_time_mask(fake_score, score_length)
                scale = gan_loss_scale / (
                    fake_score.size()[1:-1].numel() * score_length.sum()
                ).clamp_min(1.0)

                as_real_data_losses = functional.lsgan_as_real_data_loss(
                    fake_score, reduction="none"
                )
                as_real_data_loss = scale * (as_real_data_losses * score_mask).sum()

            loss_map[f"G/GAN/as_real_data_loss/{name}"] = as_real_data_loss

            if self._with_feature_matching_loss:
                node_name_to_time_stride = (
                    self._spec.feature_matching_node_name_to_time_stride
                )
                if length is not None and node_name_to_time_stride is None:
                    raise ValueError(
                        "``GANSpec.feature_matching_node_name_to_time_stride`` must be "
                        "given."
                    )

                feature_matching_losses = []
                for node_name in self._spec.feature_matching_node_names:
                    fake_feature = node_name_to_fake_feature[node_name]
                    real_feature = node_name_to_real_feature[node_name]

                    if length is None:
                        feature_matching_losses.append(
                            node_name_to_feature_matching_loss_scale[node_name]
                            * F.l1_loss(fake_feature, real_feature)
                        )
                    else:
                        time_stride = node_name_to_time_stride[node_name]
                        feature_length = length // time_stride
                        mask = _compute_time_mask(fake_feature, feature_length)
                        scale = node_name_to_feature_matching_loss_scale[node_name] / (
                            fake_feature.size()[1:-1].numel() * feature_length.sum()
                        ).clamp_min(1.0)

                        feature_matching_losses.append(
                            scale
                            * (
                                F.l1_loss(fake_feature, real_feature, reduction="none")
                                * mask
                            ).sum()
                        )

                loss_map[f"{name}/G/feature_matching_loss"] = torch.stack(
                    feature_matching_losses
                ).sum()

        return loss_map


class GANTaskBase(bases.TaskBase):
    def __init__(
        self,
        name_to_gan_spec_conf: Mapping[str, Mapping[str, Any]],
        optimizer_conf: Mapping[str, Any],
        scheduler_configuration_conf: Mapping[str, Any],
        loss_confs: Sequence[Mapping[str, Any]] = (),
        val_name_to_metric_conf_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
        discriminator_optimizer_conf: Optional[Mapping[str, Any]] = None,
        discriminator_scheduler_configuration_conf: Optional[Mapping[str, Any]] = None,
        gradient_clip_val: Optional[float] = None,
        discriminator_gradient_clip_val: Optional[float] = None,
        enable_discriminator_step: int = 0,
        discriminator_warmup_num_steps: int = 0,
        learning_rate_scheduler_step_interval: int = 0,
    ):
        super().__init__(
            optimizer_conf,
            scheduler_configuration_conf,
            loss_confs=loss_confs,
            val_name_to_metric_conf_map=val_name_to_metric_conf_map,
        )

        self.save_hyperparameters()

        self.automatic_optimization = False

        self.name_to_gan_heads = nn.ModuleDict(
            modules={
                name: _GANHead(hydra_utils.instantiate({**conf, "_recursive_": False}))
                for name, conf in name_to_gan_spec_conf.items()
            }
        )

    def configure_optimizers(self):
        discriminator_params = list(
            itertools.chain.from_iterable(
                gan_head.discriminator.parameters()
                for gan_head in self.name_to_gan_heads.values()
            )
        )
        discriminator_params_set = set(discriminator_params)
        generator_params = [
            p for p in self.parameters() if p not in discriminator_params_set
        ]

        (
            optimizer,
            scheduler_configuration,
        ) = utils.build_optimizer_and_scheduler_configuration(
            generator_params,
            self.hparams.optimizer_conf,
            self.hparams.scheduler_configuration_conf,
        )
        optimizers = [optimizer]
        scheduler_configurations = [scheduler_configuration]

        if discriminator_params:
            discriminator_optimizer_conf = (
                self.hparams.optimizer_conf
                if self.hparams.discriminator_optimizer_conf is None
                else self.hparams.discriminator_optimizer_conf
            )
            discriminator_scheduler_configuration_conf = (
                self.hparams.scheduler_configuration_conf
                if self.hparams.discriminator_scheduler_configuration_conf is None
                else self.hparams.discriminator_scheduler_configuration_conf
            )
            (
                discriminator_optimizer,
                discriminator_scheduler_configuration,
            ) = utils.build_optimizer_and_scheduler_configuration(
                discriminator_params,
                discriminator_optimizer_conf,
                discriminator_scheduler_configuration_conf,
            )

            optimizers.append(discriminator_optimizer)
            scheduler_configurations.append(discriminator_scheduler_configuration)

        for scheduler_configuration in scheduler_configurations:
            if scheduler_configuration.keys() & {"interval", "step"}:
                raise NotImplementedError

        return optimizers, scheduler_configurations

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int):
        super().on_before_optimizer_step(optimizer, optimizer_idx)

        gradient_clip_val = (
            self.hparams.gradient_clip_val
            if optimizer_idx == 0
            else self.hparams.discriminator_gradient_clip_val
        )
        if gradient_clip_val is not None:
            self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val)

    def training_step(self, batch, batch_idx):
        del batch_idx  # Unused.

        output = self(batch)

        match self.optimizers():
            case optimizer_g, optimizer_d:
                global_step = self.trainer.fit_loop.epoch_loop.total_batch_idx
                if global_step >= self.hparams.enable_discriminator_step:
                    self._train_discriminator(batch, output, optimizer_d, 1)

            case optimizer_g:
                pass

        self._train_generator(batch, output, optimizer_g, 0)

        global_step = self.trainer.fit_loop.epoch_loop.total_batch_idx
        if (global_step + 1) % self.hparams.learning_rate_scheduler_step_interval == 0:
            for scheduler in self.lr_schedulers():
                scheduler.step()

    def _maybe_compute_discriminator_loss(self, output, batch, log_scope=None):
        loss_map = {}
        for name, gan_head in self.name_to_gan_heads.items():
            gan_head_loss_map = gan_head.maybe_compute_discriminator_loss(output, batch)
            loss_map.update(
                {f"{name}/{key}": value for key, value in gan_head_loss_map.items()}
            )

        if log_scope is not None:
            self.log_dict(
                {f"{log_scope}/{key}": value for key, value in loss_map.items()}
            )

        return loss_map

    def _compute_generator_loss(self, output, batch, log_scope=None):
        if self.trainer.fit_loop.epoch_loop.total_batch_idx < (
            self.hparams.enable_discriminator_step
            + self.hparams.discriminator_warmup_num_steps
        ):
            return self.compute_loss(output, batch, log_scope=log_scope)

        loss_map = {}
        for name, gan_head in self.name_to_gan_heads.items():
            gan_head_loss_map = gan_head.maybe_compute_generator_loss(output, batch)
            loss_map.update(
                {f"{name}/{key}": value for key, value in gan_head_loss_map.items()}
            )

        if log_scope is not None:
            self.log_dict(
                {f"{log_scope}/{key}": value for key, value in loss_map.items()}
            )

        return torch.stack(
            [self.compute_loss(output, batch, log_scope=log_scope), *loss_map.values()]
        ).sum()

    def _backward_and_step(self, loss, optimizer, optimizer_index):
        precision_plugin = self.trainer.strategy.precision_plugin
        with contextlib.ExitStack() as stack:
            if isinstance(
                precision_plugin, precisions.MultipleGradScalersMixedPrecisionPlugin
            ):
                stack.enter_context(precision_plugin.with_grad_scaler(optimizer_index))

            self.manual_backward(loss)
            optimizer.step()

    def _train_discriminator(self, batch, output, optimizer, optimizer_index):
        loss_map = self._maybe_compute_discriminator_loss(
            output, batch, log_scope="train"
        )
        if not loss_map:
            return

        loss = torch.stack(list(loss_map.values())).sum()

        optimizer.zero_grad()
        self._backward_and_step(loss, optimizer, optimizer_index)

        self.log("train/discriminator_loss", loss, prog_bar=True)

    def _train_generator(self, batch, output, optimizer, optimizer_index):
        loss = self._compute_generator_loss(output, batch, log_scope="train")

        optimizer.zero_grad()
        self._backward_and_step(loss, optimizer, optimizer_index)

        self.log("train/generator_loss", loss, prog_bar=True)
