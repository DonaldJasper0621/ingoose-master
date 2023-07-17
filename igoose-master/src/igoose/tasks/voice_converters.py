import os
from typing import Any, Mapping, Optional, Sequence

from hydra import utils
from torch import nn
from torch.cuda import amp

from igoose import data_map as dm
from igoose.ios import task_ios
from igoose.nn import resamplings
from igoose.tasks import gan_bases


class GANVoiceConverter(gan_bases.GANTaskBase):
    def __init__(
        self,
        sample_rate: int,
        global_style_feature_extractor_input_sample_rate: int,
        global_style_feature_extractor_conf: Mapping[str, Any],
        voice_synthesizer_conf: Mapping[str, Any],
        pretrained_decoder_checkpoint_path: str | os.PathLike,
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

        self.style_signal_resample = None
        if global_style_feature_extractor_input_sample_rate is not None:
            self.style_signal_resample = resamplings.KaiserBestResample(
                sample_rate, global_style_feature_extractor_input_sample_rate
            )

        self.global_style_feature_extractor = utils.instantiate(
            global_style_feature_extractor_conf
        )
        self.voice_synthesizer = utils.instantiate(voice_synthesizer_conf)

        self._pretrained_module_map: dict[str, nn.Module] = {
            "decoder": task_ios.load_task(
                utils.to_absolute_path(pretrained_decoder_checkpoint_path)
            )
        }

        self._logged_validation_audios_counter = 0

    def on_validation_start(self):
        self._logged_validation_audios_counter = 0

        return super().on_validation_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return super().validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        with amp.autocast(enabled=False):
            style_signal = self.style_signal_resample(batch[dm.SIGNAL])

        global_style = self.global_style_feature_extractor(
            {**batch, dm.SIGNAL: style_signal}
        )[dm.GLOBAL_STYLE]

        return self.voice_synthesizer({**batch, dm.GLOBAL_STYLE: global_style})
