import pathlib
from typing import Sequence

import hydra
from hydra import utils
import omegaconf
import soundfile as sf
import torch
from torch import cuda
import torchaudio
import tqdm

from igoose import data_map as dm


def _save(obj, f: pathlib.Path) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    return torch.save(obj, f)


def _get_output_pt_path(
    input_audio_path: pathlib.Path, output_name_suffix: str
) -> pathlib.Path:
    return input_audio_path.with_suffix(f".{output_name_suffix}.pt")


def compute_log_mel_and_loudness(
    input_audio_paths: Sequence[pathlib.Path], cfg: omegaconf.DictConfig
):
    log_mel_spectrogram = utils.instantiate(cfg.log_mel_spectrogram_conf)
    log_mel_spectrogram.eval()

    for input_audio_path in tqdm.tqdm(
        input_audio_paths, desc="compute log mel and loudness"
    ):
        x, _ = torchaudio.load(input_audio_path)

        output_map = log_mel_spectrogram({dm.SIGNAL: x})

        log_mel = output_map[dm.CODE]
        loudness = output_map[dm.LOUDNESS]

        _save(
            log_mel.cpu(),
            _get_output_pt_path(input_audio_path, cfg.log_mel_name_suffix),
        )
        _save(
            loudness.cpu(),
            _get_output_pt_path(input_audio_path, cfg.loudness_name_suffix),
        )


def compute_f0(input_audio_paths: Sequence[pathlib.Path], cfg: omegaconf.DictConfig):
    f0_estimator = utils.instantiate(cfg.f0_estimator_conf)
    f0_estimator.eval()

    for input_audio_path in tqdm.tqdm(input_audio_paths, desc="compute f0"):
        x, _ = torchaudio.load(input_audio_path)

        f0 = f0_estimator({dm.SIGNAL: x[None]})[dm.F0].squeeze(dim=0)

        _save(
            f0.cpu(),
            _get_output_pt_path(input_audio_path, cfg.f0_name_suffix),
        )


def compute_frame_content(
    input_audio_paths: Sequence[pathlib.Path], cfg: omegaconf.DictConfig
):
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    frame_content_feature_extractor = utils.instantiate(
        cfg.frame_content_feature_extractor_conf
    )
    frame_content_feature_extractor.to(device)
    frame_content_feature_extractor.eval()

    for input_audio_path in tqdm.tqdm(input_audio_paths, desc="compute frame content"):
        x, _ = torchaudio.load(input_audio_path)
        x = x.to(device)

        frame_content = frame_content_feature_extractor({dm.SIGNAL: x[None]})[
            dm.FRAME_CONTENT
        ].squeeze(dim=0)

        _save(
            frame_content.cpu(),
            _get_output_pt_path(input_audio_path, cfg.frame_content_name_suffix),
        )


def compute_speaker_embedding(
    input_audio_paths: Sequence[pathlib.Path], cfg: omegaconf.DictConfig
):
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    speaker_embedding_model = utils.instantiate(cfg.speaker_embedding_model_conf)
    speaker_embedding_model.to(device)
    speaker_embedding_model.eval()

    for input_audio_path in tqdm.tqdm(
        input_audio_paths, desc="compute speaker_embedding"
    ):
        x, _ = torchaudio.load(input_audio_path)
        x = x.to(device)

        speaker_embedding = speaker_embedding_model({dm.SIGNAL: x[None]})[
            dm.SPEAKER_EMBEDDING
        ].squeeze(dim=0)

        _save(
            speaker_embedding.cpu(),
            _get_output_pt_path(input_audio_path, cfg.speaker_embedding_name_suffix),
        )


@hydra.main(config_path=None, config_name="precompute_features", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    input_audio_paths = sorted(
        pathlib.Path(cfg.audio_root_folder).rglob(cfg.audio_rglob_pattern)
    )

    sample_rate = cfg.sample_rate
    for input_audio_path in tqdm.tqdm(
        input_audio_paths, desc="check sample rate and num channels"
    ):
        info = sf.info(input_audio_path)

        if info.samplerate != sample_rate:
            raise ValueError(
                f"The sample rate of {input_audio_path} is {info.samplerate} instead of"
                f" {sample_rate}."
            )

        if info.channels != 1:
            raise NotImplementedError

    stage_to_fn = {
        fn.__name__: fn
        for fn in [
            compute_log_mel_and_loudness,
            compute_f0,
            compute_frame_content,
            compute_speaker_embedding,
        ]
    }

    with torch.inference_mode():
        for stage in cfg.stages:
            fn = stage_to_fn[stage]
            fn(input_audio_paths, cfg)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
