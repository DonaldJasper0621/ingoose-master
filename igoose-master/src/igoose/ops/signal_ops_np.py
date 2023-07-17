from numpy import typing as npt
import numpy as np
import parselmouth


def compute_f0_parselmouth(
    single_channel_audio: npt.NDArray,
    sample_rate: int,
    hop_length: int,
    f0_floor: float,
    f0_ceiling: float,
    silence_threshold: float = 0.03,
    voicing_threshold: float = 0.45,
    very_accurate: bool = False,
) -> npt.NDArray:
    if single_channel_audio.ndim != 1:
        raise NotImplementedError

    half_pitch_ac_window_size = int(
        int((6 if very_accurate else 3) / f0_floor * sample_rate) / 2 - 1
    )
    # We use the output length of ``librosa.stft`` with even ``n_fft`` and
    # ``center=True`` as the output length of f0.
    expected_output_f0_length = single_channel_audio.size // hop_length + 1
    # We pad hop_length samples on the right side to make sure that the f0 length
    # returned by ``to_pitch_ac`` is always greater than or equal to
    # ``expected_output_f0_length``.
    # See also https://github.com/praat/praat/issues/2011.
    padded_sound = parselmouth.Sound(
        np.pad(
            single_channel_audio,
            [half_pitch_ac_window_size, half_pitch_ac_window_size + hop_length],
            mode="reflect",
        ).astype(np.float64),
        sampling_frequency=sample_rate,
    )
    return padded_sound.to_pitch_ac(
        time_step=hop_length / sample_rate,
        pitch_floor=f0_floor,
        pitch_ceiling=f0_ceiling,
        silence_threshold=silence_threshold,
        voicing_threshold=voicing_threshold,
        very_accurate=very_accurate,
    ).selected_array["frequency"][:expected_output_f0_length]
