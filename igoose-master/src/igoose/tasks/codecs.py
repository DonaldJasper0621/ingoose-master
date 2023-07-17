import itertools
from typing import Any, Mapping, Optional, Sequence

from hydra import utils as hydra_utils
from pytorch_lightning import utilities
from torch import jit

from igoose import data_map as dm
from igoose.tasks import gan_bases


class GANCodec(gan_bases.GANTaskBase):
    def __init__(
        self,
        encoder_conf: Mapping[str, Any],
        decoder_conf: Mapping[str, Any],
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
        num_logged_validation_audios_max: int = 5,
    ):
        super().__init__(
            name_to_gan_spec_conf,
            optimizer_conf,
            scheduler_configuration_conf,
            loss_confs=loss_confs,
            val_name_to_metric_conf_map=val_name_to_metric_conf_map,
            discriminator_optimizer_conf=discriminator_optimizer_conf,
            discriminator_scheduler_configuration_conf=(
                discriminator_scheduler_configuration_conf
            ),
            gradient_clip_val=gradient_clip_val,
            discriminator_gradient_clip_val=discriminator_gradient_clip_val,
            enable_discriminator_step=enable_discriminator_step,
            discriminator_warmup_num_steps=discriminator_warmup_num_steps,
            learning_rate_scheduler_step_interval=learning_rate_scheduler_step_interval,
        )

        self.save_hyperparameters()

        self.encoder = hydra_utils.instantiate(encoder_conf)
        self.decoder = hydra_utils.instantiate(decoder_conf)

        self._logged_validation_audios_counter = 0

    def training_step(self, batch, batch_idx):
        batch = self._maybe_encode(batch)
        batch = self._maybe_use_input_signal_and_code_as_target(batch)
        return super().training_step(batch, batch_idx)

    def on_validation_start(self):
        self._logged_validation_audios_counter = 0

        return super().on_validation_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._maybe_encode(batch)
        batch = self._maybe_use_input_signal_and_code_as_target(batch)
        return super().validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def forward(
        self, batch: Mapping[str, Any], always_return_output_code: bool = True
    ) -> dict[str, Any]:
        batch = self._maybe_encode(batch)

        output_map = self.decode(batch)

        input_signal = batch.get(dm.SIGNAL)
        if input_signal is not None:
            output_map[dm.SIGNAL] = output_map[dm.SIGNAL][
                ..., : input_signal.size(dim=-1)
            ]

        signal_length_key = dm.get_length_key(dm.SIGNAL)
        if signal_length_key in output_map:
            input_signal_length = batch.get(signal_length_key)
            if input_signal_length is not None:
                output_map[signal_length_key] = input_signal_length.clone()

        if always_return_output_code and dm.CODE not in output_map:
            encoder_output_map = self.encode(
                {**batch, dm.SIGNAL: output_map[dm.SIGNAL]}
            )
            for key, value in encoder_output_map.items():
                output_map.setdefault(key, value)

        return output_map

    def _maybe_encode(self, batch):
        batch = dict(batch)

        if dm.CODE not in batch:
            encoder_output_map = self.encode(batch)

            batch[dm.CODE] = encoder_output_map[dm.CODE]

            code_length = encoder_output_map.get(dm.get_length_key(dm.CODE))
            if code_length is not None:
                batch.setdefault(dm.get_length_key(dm.CODE), code_length)

        return batch

    def _maybe_use_input_signal_and_code_as_target(self, batch):
        del self  # Unused.

        batch = dict(batch)

        for input_key, target_key in [
            (dm.SIGNAL, dm.TARGET_SIGNAL),
            (dm.CODE, dm.TARGET_CODE),
        ]:
            batch.setdefault(target_key, batch[input_key])

            input_length = batch.get(dm.get_length_key(input_key))
            if input_length is not None:
                batch.setdefault(dm.get_length_key(target_key), input_length)

        return batch

    @utilities.rank_zero_only
    def _add_validation_summaries(self, output_map, batch):
        if (
            self._logged_validation_audios_counter
            >= self.hparams.num_logged_validation_audios_max
        ):
            return

        if self.logger is None:
            return

        tensorboard = self.logger.experiment
        global_step = self.trainer.fit_loop.epoch_loop.total_batch_idx
        for input_signal, input_sample_rate, input_length, output_signal in zip(
            batch[dm.SIGNAL],
            batch[dm.SAMPLE_RATE].tolist(),
            batch.get(dm.get_length_key(dm.SIGNAL), itertools.repeat(None)),
            output_map[dm.SIGNAL],
        ):
            input_signal = input_signal[..., :input_length]
            output_signal = output_signal[..., :input_length]

            if not self.trainer.sanity_checking:
                index = self._logged_validation_audios_counter
                tensorboard.add_audio(
                    f"validation/item_{index}/input",
                    input_signal.squeeze(dim=0),
                    global_step=global_step,
                    sample_rate=input_sample_rate,
                )
                tensorboard.add_audio(
                    f"validation/item_{index}/output",
                    output_signal.squeeze(dim=0),
                    global_step=global_step,
                    sample_rate=input_sample_rate,
                )

            self._logged_validation_audios_counter += 1

            if (
                self._logged_validation_audios_counter
                >= self.hparams.num_logged_validation_audios_max
            ):
                break

    @jit.export
    def encode(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return self.encoder(batch)

    @jit.export
    def decode(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return self.decoder(batch)
