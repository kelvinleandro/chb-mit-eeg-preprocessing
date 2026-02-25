import numpy as np
from feature_extractor.base import FeatureExtractor


class CovarianceExtractor(FeatureExtractor):
    def extract(signal: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """
        Extract covariance-based features from a multi-channel signal.

        This method computes the channel-wise covariance matrix of the input
        signal (using channels as variables and samples as observations).
        The diagonal elements of the covariance matrix are scaled by sqrt(2)
        to preserve the Frobenius norm when vectorizing the symmetric matrix.

        The upper triangular part of the covariance matrix (including the
        diagonal) is then flattened into a column vector of shape
        (n_features, 1).

        Notes:
            - The input signal must have shape (n_channels, n_samples).
            - The sample_rate parameter is accepted for API compatibility
              but is not used in this extractor.
            - If n_channels = C, the number of output features is
              C * (C + 1) // 2.

        Args:
            signal (np.ndarray): Multi-channel signal of shape
                (n_channels, n_samples).
            sample_rate (int | None, optional): Sampling rate of the signal.
                Not used in this implementation.

        Returns:
            np.ndarray: Column vector containing the vectorized upper
            triangular covariance values with shape (n_features, 1).
        """
        cov_matrix = np.cov(signal)

        # Multiply the diagonal values by sqrt(2)
        np.fill_diagonal(cov_matrix, np.diag(cov_matrix) * np.sqrt(2))

        # Vectorize the upper triangular as (n_features, 1)
        upper_triangular = cov_matrix[np.triu_indices_from(cov_matrix)].reshape(-1, 1)

        return upper_triangular
