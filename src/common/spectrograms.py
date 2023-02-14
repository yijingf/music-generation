# Implement ddsp.spectral_ops in numpy
# FFT operations

import librosa
import numpy as np


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def safe_log(x, eps=1e-5):
    idx = np.where(x <= 0)
    x[idx] = eps
    return np.log(x)


def stft(audio, frame_size=2048, hop_size=128, pad_end=True):
    """Non-differentiable stft using librosa, one example at a time."""

    if pad_end:
        audio = pad(audio, frame_size, hop_size)

    s = librosa.stft(y=audio,
                     n_fft=int(frame_size),
                     hop_length=hop_size,
                     center=False).T
    return s


def compute_logmel(audio,
                   lo_hz=20.0,
                   hi_hz=7600.0,
                   num_mel_bins=512,
                   fft_size=2048,
                   hop_size=128,
                   pad_end=True,
                   sample_rate=16000):
    """Logarithmic amplitude of mel-scaled spectrogram."""
    mag = np.abs(stft(audio, fft_size, hop_size, pad_end))
    num_spectrogram_bins = int(mag.shape[-1])

    linear_to_mel_matrix = linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)

    mel = np.dot(mag, linear_to_mel_matrix)
    return safe_log(mel)


def pad(x, frame_size, hop_size):
    """ Pad 0 to the end of tensor such that n_frames = n_t / hop_size. 

    Args:
      x: Tensor to pad, any shape.
      frame_size: Size of frames for striding.
      hop_size: Striding, space between frames.

    Returns:
      A padded version of `x` along axis. Output sizes can be computed separately
        with strided_lengths.
    """
    if hop_size > frame_size:
        raise ValueError(f'During padding, frame_size ({frame_size})'
                         f' must be greater than hop_size ({hop_size}).')

    n_t = x.size
    n_frames = int(np.ceil(n_t / hop_size))
    n_t_padded = (n_frames - 1) * hop_size + frame_size

    x_padded = np.pad(x, [0, int(n_t_padded - n_t)])

    return x_padded


def _hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.
    Args:
      frequencies_hertz: An array of frequencies in Hertz.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=8000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0,):
    """Returns a matrix to warp linear scale spectrograms to the [mel scale][mel].
    """

    # HTK excludes the spectrogram DC bin.
    f_nyquist = sample_rate / 2.0

    # Linear frequencies
    linear_frequencies = np.linspace(0, f_nyquist, num_spectrogram_bins)[1:]
    spectrogram_bins_mel = np.expand_dims(
        _hertz_to_mel(linear_frequencies.astype(np.float32)), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.

    edge_mel = np.linspace(_hertz_to_mel(lower_edge_hertz),
                           _hertz_to_mel(upper_edge_hertz),
                           num_mel_bins + 2).astype(np.float32)

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = (
        edge_mel[i:i + num_mel_bins] for i in range(3))

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    return np.pad(mel_weights_matrix, [[1, 0], [0, 0]])
