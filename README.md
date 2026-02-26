# EEG Preprocessing for CHB-MIT Scalp EEG Database

This project provides a pipeline for preprocessing the **CHB-MIT Scalp EEG database** from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/). It handles EDF file loading, extraction of ictal and pre-ictal segments, epoching, and feature extraction using covariance-based methods.

## Features

- **Segment Extraction:**
  - **Ictal:** Automatically identifies and extracts seizure segments based on the patient-specific summary files.
  - **Pre-ictal:** Extracts segments preceding seizures with a configurable offset and duration multiplier.
- **Epoching:** Splits signal segments into non-overlapping fixed-length epochs (default: 5 seconds).
- **Feature Extraction:** Implements a `CovarianceExtractor` that computes channel-wise covariance matrices and vectorizes them, preserving the Frobenius norm.
- **Output:** Saves processed features and labels (`1` for ictal, `0` for pre-ictal) as compressed `.npz` files for each patient in the `out/data/` directory.

## Prerequisites

- Python >= 3.12
- Dependencies: `mne`, `numpy`

## Installation

You can install the dependencies using `pip` or a package manager like `uv`:

```bash
pip install mne numpy
```

Or if you are using `uv`:

```bash
uv sync
```

## Dataset Structure

The project expects the CHB-MIT Scalp EEG database to be organized in its original structure as downloaded from PhysioNet. Point the `PATH_ROOT_DATASET` in `constants.py` to this root folder or pass it as a command-line argument.

```text
chb-mit-scalp-eeg-database/
├── RECORDS-WITH-SEIZURES
├── chb01/
│   ├── chb01-summary.txt
│   ├── chb01_01.edf
│   ├── chb01_02.edf
│   └── ...
├── chb02/
│   ├── chb02-summary.txt
│   ├── chb02_01.edf
│   └── ...
└── ...
```

## Usage

To run the preprocessing pipeline, execute the `main.py` script. The script uses command-line arguments to configure the pipeline:

```bash
python main.py [options]
```

### Available Arguments

- `--path`, `-p`: Root directory of the EEG dataset (default: defined in `constants.py`).
- `--offset_seconds`, `-o`: Time gap (in seconds) between the pre-ictal segment end and the seizure onset (default: 300).
- `--multiplier`, `-m`: Factor used to scale the pre-ictal segment duration relative to the seizure length (default: 3).
- `--epoch_duration`, `-e`: Duration of each signal epoch in seconds for feature extraction (default: 5).

### Example

```bash
python main.py -p "path/to/dataset" -o 60 -m 5 -e 2
```

The script will:

1. Read the `RECORDS-WITH-SEIZURES` file from the dataset root.
2. Process each record by identifying seizure times from summary files.
3. Extract ictal and pre-ictal segments.
4. Split segments into fixed-length epochs.
5. Extract covariance-based features.
6. Save the results as compressed `.npz` files in `out/data/<patient_id>.npz`.

## Project Structure

- `main.py`: Main entry point for the preprocessing pipeline.
- `constants.py`: Configuration for dataset paths and channel selections.
- `edf.py`: Module for reading EDF files and parsing seizure metadata.
- `signals.py`: Utility functions for segment extraction and epoching.
- `train_test_split.py`: Utility to load and split patient data into train/test sets.
- `feature_extractor/`:
  - `base.py`: Abstract base class for feature extractors.
  - `covariance.py`: Implementation of covariance-based feature extraction.
- `out/`: Directory where logs and processed data are stored.
