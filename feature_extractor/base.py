from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor(ABC):

    @classmethod
    @abstractmethod
    def extract(cls, signal: np.ndarray, sample_rate: int | None) -> np.ndarray:
        """
        Extract features from a signal.

        Args:
            signal (np.ndarray): The input signal of shape (n_channels, n_samples).
            sample_rate (int, optional): The sample rate of the signal.

        Returns:
            np.ndarray: The extracted features.
        """

    @classmethod
    def extract_all(
        cls,
        signals: list[np.ndarray] | np.ndarray[np.ndarray],
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """
        Extract features from a list of signals.

        Args:
            signals (list[np.ndarray] | np.ndarray[np.ndarray]): The list of input signals.

            sample_rate (int, optional): The sample rate of the signals.

        Returns:
            np.ndarray: The extracted features.
        """

        if isinstance(signals, np.ndarray):
            if signals.ndim == 2:
                signals = [signals]
            else:
                raise ValueError(
                    "'signals' must be a list of 2D numpy arrays or a 3D numpy array with shape (n_signals, n_channels, n_samples)."
                )

        return np.array([cls.extract(signal, sample_rate) for signal in signals])
