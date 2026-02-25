import logging
import os
from datetime import datetime

os.makedirs("out", exist_ok=True)

file_name = datetime.now().strftime("log_%Y-%m-%d.log")
path_log = os.path.join("out", file_name)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(path_log, encoding="utf-8"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Hello from eeg-preprocessing!")


if __name__ == "__main__":
    main()
