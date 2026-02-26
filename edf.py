import numpy as np
from mne.io import read_raw_edf
from pathlib import Path
import re
from signals import SignalSegment
from constants import CHANNELS_TO_KEEP


class EDF:
    def __init__(self, path: Path | str, with_seizures: bool = False) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.seizures: list[SignalSegment] = []

        self._read()

        if with_seizures:
            self._load_seizures()

    def _read(self) -> None:
        with read_raw_edf(self.path, preload=False, verbose=False) as raw:
            self.sample_rate = int(raw.info["sfreq"])
            self.channel_names = raw.ch_names
            if "T8-P8-0" in self.channel_names:
                # Some files there are 2 'T8-P8'
                self.channel_names[self.channel_names.index("T8-P8-0")] = "T8-P8"

            idx_map = {v: i for i, v in enumerate(self.channel_names)}
            indices = np.fromiter(
                (idx_map[x] for x in CHANNELS_TO_KEEP if x in idx_map), dtype=int
            )

            if len(indices) != len(CHANNELS_TO_KEEP):
                raise ValueError(
                    f"The following channels does not exist in '{self.path.name}': {set(CHANNELS_TO_KEEP) - set(self.channel_names)}"
                )

            self.data = np.take(raw.get_data(), indices, axis=0)

    def _load_seizures(self) -> None:
        patient = self.path.name.split("_")[0]
        parent = self.path.parent
        summary = parent / f"{patient}-summary.txt"
        with open(summary, "r") as f:
            content = f.read()
            _start = content.find(f"File Name: {self.path.name}")
            _end = content.find("File Name", _start + 1)
            content = content[_start:_end]
            starts = re.findall(r"Start Time: (\d+)\sseconds", content)
            ends = re.findall(r"End Time: (\d+)\sseconds", content)
            seizures = [
                SignalSegment(
                    int(start) * self.sample_rate, int(end) * self.sample_rate
                )
                for start, end in zip(starts, ends)
            ]
            self.seizures = seizures

    def get_seizure_data(self) -> list[np.ndarray]:
        return [self.data[:, seg.start : seg.end] for seg in self.seizures]
