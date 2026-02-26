import numpy as np
from typing import Tuple, List
from pathlib import Path


def load_train_test_split(
    data_dir: str | Path,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset by patient (one .npz file per patient).

    Args:
    data_dir : str
        Path to directory containing .npz files.
    train_ratio : float
        Ratio of patient used for training (0 < train_ratio < 1).
    shuffle : bool
        Whether to shuffle patient before splitting.
    random_state : int
        Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Directory '{data_dir}' does not exist")

    if not data_dir.is_dir():
        raise ValueError(f"'{data_dir}' is not a directory")

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")

    files = [f for f in data_dir.iterdir() if f.suffix == ".npz"]
    files.sort()

    if len(files) == 0:
        raise ValueError(f"No .npz files found in '{data_dir}'")

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(files)

    split_idx = int(len(files) * train_ratio)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    if len(train_files) == 0 or len(test_files) == 0:
        raise ValueError("No files found for training or testing")

    def load_files(file_list: List[str]):
        X_list, y_list = [], []
        for fname in file_list:
            data = np.load(fname)
            X_list.append(data["features"])
            y_list.append(data["labels"])
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_train, y_train = load_files(train_files)
    X_test, y_test = load_files(test_files)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_test_split(data_dir="out/data")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
