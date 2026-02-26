from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalSegment:
    start: int
    "Start index of the segment."
    end: int
    "End index of the segment."


def get_pre_ictal_segment(
    signal: np.ndarray,
    start_idx: int,
    end_idx: int,
    sample_rate: int = 256,
    offset_seconds: int = 60,
    multiplier: int = 1,
) -> np.ndarray:
    """
    Extract a pre-ictal segment preceding an ictal event.

    The duration of the ictal event is first computed as:
        L = end_idx - start_idx

    A pre-ictal window is then defined such that:
        - It ends `offset_seconds` before the ictal onset (`start_idx`).
        - Its duration is `multiplier * L`, i.e., a multiple of the ictal
          event duration.

    If the computed window exceeds the signal boundaries (i.e., negative
    indices), the start and stop indices are clipped to 0.

    The input signal is assumed to have shape (n_channels, n_samples).

    Args:
        signal (np.ndarray): Multi-channel signal array with shape
            (n_channels, n_samples).
        start_idx (int): Sample index corresponding to the ictal onset.
        end_idx (int): Sample index corresponding to the ictal end.
        sample_rate (int, optional): Sampling rate in Hz. Defaults to 256.
        offset_seconds (int, optional): Time gap (in seconds) between the
            end of the pre-ictal segment and the ictal onset. Defaults to 60.
        multiplier (int, optional): Factor used to scale the ictal duration
            when defining the length of the pre-ictal segment. Defaults to 1.

    Returns:
        np.ndarray: Pre-ictal segment with shape
        (n_channels, n_pre_ictal_samples). May be shorter than expected
        if clipped by signal boundaries.
    """
    if signal.ndim != 2:
        raise ValueError("Signal must have shape (n_channels, n_samples)")

    L = end_idx - start_idx
    offset_samples = int(offset_seconds * sample_rate)
    stop_point = start_idx - offset_samples
    start_point = stop_point - (L * multiplier)

    final_start = max(0, start_point)
    final_stop = max(0, stop_point)

    return signal[:, final_start:final_stop]


def get_epochs(
    signal: np.ndarray, sample_rate: int = 256, epoch_duration: int = 5
) -> list[np.ndarray]:
    """
    Split a multi-channel signal into non-overlapping fixed-length epochs.

    The input signal is assumed to have shape (n_channels, n_samples).

    The number of epochs is computed using floor division:

        n_epochs = n_samples // (sample_rate * epoch_duration)

    Only complete epochs are returned. Any remaining samples that do not
    form a full epoch are discarded.

    Example:
        - n_samples = 2560, sample_rate = 256, epoch_duration = 2
            → epoch_size = 512 → 2560 // 512 = 5 epochs

    Args:
        signal (np.ndarray): Multi-channel signal array with shape
            (n_channels, n_samples).
        sample_rate (int, optional): Sampling rate in Hz. Defaults to 256.
        epoch_duration (int, optional): Duration of each epoch in seconds.
            Defaults to 5.

    Returns:
        np.ndarray: Epochs array with shape (n_epochs, n_channels, epoch_size).
    """
    if signal.ndim != 2:
        raise ValueError("Signal must have shape (n_channels, n_samples)")

    n_channels, n_samples = signal.shape
    epoch_size = sample_rate * epoch_duration
    n_epochs = n_samples // epoch_size

    if n_epochs == 0:
        logger.warning("No epochs found. Returning empty array.")
        return np.empty((0, n_channels, epoch_size), dtype=signal.dtype)

    valid_samples = n_epochs * epoch_size
    truncated_signal = signal[:, :valid_samples]

    epochs = truncated_signal.reshape(n_channels, n_epochs, epoch_size).swapaxes(0, 1)
    return epochs
