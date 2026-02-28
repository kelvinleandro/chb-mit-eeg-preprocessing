import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
from time import time
from argparse import ArgumentParser
from scipy.io import loadmat, savemat

from constants import PATH_ROOT_DATASET
from edf import EDF
from signals import get_epochs, get_pre_ictal_segment
from feature_extractor.covariance import CovarianceExtractor

warnings.filterwarnings(
    "ignore", message="Channel names are not unique*", category=RuntimeWarning
)

parser = ArgumentParser(description="EEG Feature Extraction")
parser.add_argument(
    "--path",
    "-p",
    help="Root directory of the EEG dataset",
    type=str,
    default=PATH_ROOT_DATASET,
)
parser.add_argument(
    "--offset_seconds",
    "-o",
    help="Time gap (in seconds) between the pre-ictal segment end and the seizure onset (default: %(default)s)",
    type=int,
    default=5 * 60,
)
parser.add_argument(
    "--multiplier",
    "-m",
    help="Factor to scale the pre-ictal segment duration relative to the seizure length (default: %(default)s)",
    type=int,
    default=3,
)
parser.add_argument(
    "--epoch_duration",
    "-e",
    help="Duration of each signal epoch in seconds for feature extraction (default: %(default)s)",
    type=int,
    default=5,
)
parser.add_argument(
    "--output_type",
    "-t",
    choices=["npz", "mat"],
    help="Output file type (default: %(default)s)",
    type=str,
    default="npz",
)

args = parser.parse_args()
DATASET_PATH = Path(args.path)
OFFSET_SECONDS = args.offset_seconds
MULTIPLIER = args.multiplier
EPOCH_DURATION = args.epoch_duration
OUTPUT_TYPE = args.output_type

with (DATASET_PATH / "RECORDS-WITH-SEIZURES").open() as f:
    RECORDS_WITH_SEIZURES = [line.strip() for line in f if line.strip()]


os.makedirs("out", exist_ok=True)
path_data = Path("out/data")
path_data.mkdir(exist_ok=True)
(path_data.parent / "logs").mkdir(exist_ok=True)

file_name = datetime.now().strftime("log_%Y-%m-%d.log")
path_log = os.path.join("out", "logs", file_name)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(path_log, encoding="utf-8"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# clean up all files in data folder
for file in path_data.iterdir():
    os.remove(file)


def main():
    start_time = time()
    for record in RECORDS_WITH_SEIZURES:
        patient = record.split("/")[0]
        out_patient_path = path_data / f"{patient}.{OUTPUT_TYPE}"

        path_edf = DATASET_PATH / record

        if not path_edf.exists():
            logger.warning(f"'{path_edf}' does not exist. Skipping...")
            continue

        logger.info(f"PROCESSING {record}")

        try:
            edf = EDF(path_edf, with_seizures=True)
        except Exception as e:
            logging.error(e)
            continue

        if len(edf.seizures) == 0:
            logger.info(f"{record} has no seizures. Skipping...")
            continue

        raw_seizures_data = edf.get_seizure_data()
        logger.info(f"{len(raw_seizures_data)} seizure segments found.")

        # Getting pre-ictal segments
        raw_pre_ictal = [
            get_pre_ictal_segment(
                edf.data,
                seg.start,
                seg.end,
                edf.sample_rate,
                OFFSET_SECONDS,
                MULTIPLIER,
            )
            for seg in edf.seizures
        ]

        logger.info(f"{len(raw_pre_ictal)} pre-ictal segments generated.")

        # Get an array of epochs for ictal and pre-ictal
        ictal_epochs = np.concatenate(
            [
                get_epochs(seg, edf.sample_rate, EPOCH_DURATION)
                for seg in raw_seizures_data
            ]
        )

        pre_ictal_epochs = np.concatenate(
            [get_epochs(seg, edf.sample_rate, EPOCH_DURATION) for seg in raw_pre_ictal]
        )

        if len(pre_ictal_epochs) == 0:
            logger.warning(f"{record} has no pre-ictal epochs. Skipping...")
            continue
        elif len(ictal_epochs) == 0:
            logger.warning(f"{record} has no ictal epochs. Skipping...")
            continue

        logger.info(
            f"{len(ictal_epochs)} ictal epochs of shape {ictal_epochs[0].shape} generated."
        )

        logger.info(
            f"{len(pre_ictal_epochs)} pre-ictal epochs of shape {pre_ictal_epochs[0].shape} generated."
        )

        # Extracting features
        ictal_features = CovarianceExtractor.extract_all(ictal_epochs)
        pre_ictal_features = CovarianceExtractor.extract_all(pre_ictal_epochs)
        logger.info(
            f"Extracted {len(ictal_features)} ictal features and {len(pre_ictal_features)} pre-ictal features. The shape of each feature is {ictal_features[0].shape}"
        )

        ictal_labels = np.ones(len(ictal_features))
        pre_ictal_labels = np.zeros(len(pre_ictal_features))

        features = np.concatenate([ictal_features, pre_ictal_features])
        labels = np.concatenate([ictal_labels, pre_ictal_labels])

        if OUTPUT_TYPE == "npz":
            if out_patient_path.exists():
                with np.load(out_patient_path) as old_data:
                    old_features = old_data["features"]
                    old_labels = old_data["labels"]
                    features = np.concatenate([old_features, features])
                    labels = np.concatenate([old_labels, labels])

            logger.info(
                f"Saving features with shape {features.shape} and labels with shape {labels.shape} to '{out_patient_path}'"
            )

            np.savez_compressed(
                out_patient_path,
                features=features,
                labels=labels,
            )
        elif OUTPUT_TYPE == "mat":
            if out_patient_path.exists():
                old_data = loadmat(out_patient_path)
                old_features = old_data["features"]
                old_labels = np.squeeze(old_data["labels"])
                features = np.concatenate([old_features, features])
                labels = np.concatenate([old_labels, labels])

            logger.info(
                f"Saving features with shape {features.shape} and labels with shape {labels.shape} to '{out_patient_path}'"
            )

            savemat(
                out_patient_path,
                {
                    "features": features,
                    "labels": labels,
                },
            )

    elapsed_time = int(time() - start_time)
    hh, rest = divmod(elapsed_time, 3600)
    mm, ss = divmod(rest, 60)
    logger.info(f"End of data transformation. Elapsed Time: {hh:02d}:{mm:02d}:{ss:02d}")


if __name__ == "__main__":
    main()
