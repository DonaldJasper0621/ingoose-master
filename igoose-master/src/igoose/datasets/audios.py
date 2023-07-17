import glob
import os
import pathlib
import re
from typing import Iterable, Optional, Pattern, Sequence
import wave

from lightning_utilities.core import apply_func
import numpy as np
import soundfile as sf
import torch

from igoose import data_map as dm
from igoose.datasets import bases
from igoose.ios import audio_ios
from igoose.ops import array_ops_np
from igoose.ops import signal_ops


class AudioPaths(bases.Dataset):
    def __init__(
        self,
        audio_paths: Iterable[str | os.PathLike],
        sort: bool = True,
        load_only_first_channel: bool = False,
        sample_rate: Optional[int] = None,
        valid_raw_sample_rates: Optional[Sequence[int]] = None,
        valid_length_ms_min: Optional[float] = None,
        valid_length_ms_max: Optional[float] = None,
        key_to_metadata_pt_name_suffix: Optional[str] = None,
        transforms_np=None,
        transforms=None,
        cache: bool = False,
        cache_folder=None,
        load_all_cache_files: bool = True,
    ):
        self._audio_paths = self._maybe_filter_audio_paths_by_sample_rate_and_length(
            (sorted if sort else list)(map(pathlib.Path, audio_paths)),
            valid_raw_sample_rates=valid_raw_sample_rates,
            valid_length_ms_min=valid_length_ms_min,
            valid_length_ms_max=valid_length_ms_max,
        )
        if not self._audio_paths:
            raise ValueError("Empty valid audio paths.")

        self._load_only_first_channel = load_only_first_channel
        self._sample_rate = sample_rate
        self._key_to_metadata_pt_name_suffix = dict(
            key_to_metadata_pt_name_suffix or {}
        )

        if any(
            key in self._key_to_metadata_pt_name_suffix
            for key in [dm.SIGNAL, dm.SAMPLE_RATE]
        ):
            raise ValueError(
                f"{dm.SIGNAL} and {dm.SAMPLE_RATE} are read from audio files."
            )

        super().__init__(
            transforms_np=transforms_np,
            transforms=transforms,
            cache=cache,
            cache_folder=cache_folder,
            load_all_cache_files=load_all_cache_files,
        )

    def __len__(self):
        return len(self._audio_paths)

    def _maybe_filter_audio_paths_by_sample_rate_and_length(
        self,
        audio_paths,
        valid_raw_sample_rates: Optional[Sequence[int]] = None,
        valid_length_ms_min: Optional[float] = None,
        valid_length_ms_max: Optional[float] = None,
    ):
        if all(
            arg is None
            for arg in [
                valid_raw_sample_rates,
                valid_length_ms_min,
                valid_length_ms_max,
            ]
        ):
            return audio_paths

        valid_length_ms_min = (
            -1.0 if valid_length_ms_min is None else valid_length_ms_min
        )
        valid_length_ms_max = (
            float("inf") if valid_length_ms_max is None else valid_length_ms_max
        )

        valid_paths = []
        for path, (sample_rate, length) in zip(
            audio_paths, map(self.get_sample_rate_and_length_pair, audio_paths)
        ):
            length_ms = length / sample_rate * 1000
            if not valid_length_ms_min <= length_ms <= valid_length_ms_max:
                continue

            if (
                valid_raw_sample_rates is not None
                and sample_rate not in valid_raw_sample_rates
            ):
                continue

            valid_paths.append(path)

        return valid_paths

    def get_sample_rate_and_length_pair(self, path):
        del self  # Unused.

        info = sf.info(path)
        return info.samplerate, info.frames

    def load_samples_ct_and_sample_rate(self, path):
        del self  # Unused.

        samples_tc, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return samples_tc.T, sample_rate

    def _load_maybe_cached_data(self, cached_data_index):
        audio_path = self._audio_paths[cached_data_index]
        samples, sample_rate = self.load_samples_ct_and_sample_rate(audio_path)
        if self._load_only_first_channel:
            samples = samples[:1]

        if self._sample_rate is not None and sample_rate != self._sample_rate:
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            elif samples.dtype != np.float32:
                raise NotImplementedError

            samples = signal_ops.resample_kaiser_best(
                array_ops_np.as_tensor(samples), sample_rate, self._sample_rate
            ).numpy()
            sample_rate = self._sample_rate

        data_map = {
            dm.SIGNAL: samples,
            dm.SAMPLE_RATE: np.array(sample_rate, dtype=int),
        }

        for key, name_suffix in self._key_to_metadata_pt_name_suffix.items():
            pt_path = audio_path.with_suffix(f".{name_suffix}.pt")
            metadata = torch.load(pt_path)
            metadata = apply_func.apply_to_collection(
                metadata, torch.Tensor, lambda t: t.numpy()
            )
            data_map[key] = metadata

        return data_map

    @property
    def audio_paths(self):
        return list(self._audio_paths)

    def get_data_map_np(self, item_index):
        return dict(self.get_maybe_cached_data(item_index))


class AudioFolder(AudioPaths):
    def __init__(
        self,
        root_folder: str | os.PathLike,
        input_rglob_pattern: str,
        load_only_first_channel: bool = False,
        sample_rate: Optional[int] = None,
        valid_raw_sample_rates: Optional[Sequence[int]] = None,
        valid_length_ms_min: Optional[float] = None,
        valid_length_ms_max: Optional[float] = None,
        key_to_metadata_pt_name_suffix: Optional[str] = None,
        follow_symlinks: bool = False,
        exclude_patterns: Optional[Sequence[Pattern]] = None,
        transforms_np=None,
        transforms=None,
        cache: bool = False,
        cache_folder=None,
        load_all_cache_files: bool = True,
    ):
        self.root_folder = pathlib.Path(root_folder)
        if follow_symlinks:
            audio_paths = glob.glob(
                str(self.root_folder / f"**/{input_rglob_pattern}"), recursive=True
            )
        else:
            audio_paths = list(self.root_folder.rglob(input_rglob_pattern))

        if exclude_patterns:
            exclude_patterns = list(map(re.compile, exclude_patterns))
            audio_paths = [
                path
                for path in audio_paths
                if not any(pattern.match(str(path)) for pattern in exclude_patterns)
            ]

        if not audio_paths:
            raise ValueError(f"Empty ``{type(self).__name__}({self.root_folder})``.")

        super().__init__(
            audio_paths,
            sort=True,
            load_only_first_channel=load_only_first_channel,
            sample_rate=sample_rate,
            valid_raw_sample_rates=valid_raw_sample_rates,
            valid_length_ms_min=valid_length_ms_min,
            valid_length_ms_max=valid_length_ms_max,
            key_to_metadata_pt_name_suffix=key_to_metadata_pt_name_suffix,
            transforms_np=transforms_np,
            transforms=transforms,
            cache=cache,
            cache_folder=cache_folder,
            load_all_cache_files=load_all_cache_files,
        )


class _WavMixin:
    mmap: bool

    def get_sample_rate_and_length_pair(self, path):
        del self  # Unused.
        try:
            with wave.open(str(path), mode="rb") as wav:
                return wav.getframerate(), wav.getnframes()
        except wave.Error:
            # Note that ``wave`` doesn't support ``pcm_f32le``.
            info = sf.info(path)
            return info.samplerate, info.frames

    def load_samples_ct_and_sample_rate(self, path):
        sample_rate, samples_ct = audio_ios.read_wav(path, mmap=self.mmap)
        return samples_ct, sample_rate


class WavPaths(_WavMixin, AudioPaths):
    def __init__(
        self,
        wav_paths,
        sort: bool = True,
        load_only_first_channel: bool = False,
        mmap: bool = False,
        sample_rate: Optional[int] = None,
        valid_raw_sample_rates: Optional[Sequence[int]] = None,
        valid_length_ms_min: Optional[float] = None,
        valid_length_ms_max: Optional[float] = None,
        key_to_metadata_pt_name_suffix: Optional[str] = None,
        transforms_np=None,
        transforms=None,
        cache: bool = False,
        cache_folder=None,
        load_all_cache_files: bool = True,
    ):
        self.mmap = mmap

        super().__init__(
            wav_paths,
            sort=sort,
            load_only_first_channel=load_only_first_channel,
            sample_rate=sample_rate,
            valid_raw_sample_rates=valid_raw_sample_rates,
            valid_length_ms_min=valid_length_ms_min,
            valid_length_ms_max=valid_length_ms_max,
            key_to_metadata_pt_name_suffix=key_to_metadata_pt_name_suffix,
            transforms_np=transforms_np,
            transforms=transforms,
            cache=cache,
            cache_folder=cache_folder,
            load_all_cache_files=load_all_cache_files,
        )


class WavFolder(_WavMixin, AudioFolder):
    def __init__(
        self,
        root_folder,
        input_rglob_pattern="*.wav",
        load_only_first_channel: bool = False,
        mmap: bool = False,
        sample_rate: Optional[int] = None,
        valid_raw_sample_rates: Optional[Sequence[int]] = None,
        valid_length_ms_min: Optional[float] = None,
        valid_length_ms_max: Optional[float] = None,
        key_to_metadata_pt_name_suffix: Optional[str] = None,
        follow_symlinks: bool = False,
        exclude_patterns: Optional[Sequence[Pattern]] = None,
        transforms_np=None,
        transforms=None,
        cache: bool = False,
        cache_folder=None,
        load_all_cache_files: bool = True,
    ):
        self.mmap = mmap

        super().__init__(
            root_folder,
            input_rglob_pattern=input_rglob_pattern,
            load_only_first_channel=load_only_first_channel,
            sample_rate=sample_rate,
            valid_raw_sample_rates=valid_raw_sample_rates,
            valid_length_ms_min=valid_length_ms_min,
            valid_length_ms_max=valid_length_ms_max,
            key_to_metadata_pt_name_suffix=key_to_metadata_pt_name_suffix,
            follow_symlinks=follow_symlinks,
            exclude_patterns=exclude_patterns,
            transforms_np=transforms_np,
            transforms=transforms,
            cache=cache,
            cache_folder=cache_folder,
            load_all_cache_files=load_all_cache_files,
        )
