from pathlib import Path

from src.utils.urls import (
    download_file,
    DATASET_TID2008_URL,
    DATASET_TID2013_URL,
    DATASET_KADID10K_URL
)
from src.utils.paths import (
    DATASET_TID2008_ARCHIVE_PATH,
    DATASET_TID2013_ARCHIVE_PATH,
    DATASET_KADID10K_ARCHIVE_PATH
)


def download_dataset(file_url: str, destination_file_path: Path):
    if not destination_file_path.exists():
        print(f"\nRozpoczynam pobieranie datasetu: {file_url}")

        download_file(
            file_url=file_url,
            destination_file_path=destination_file_path
        )

        print(f"Dataset zapisano w: {destination_file_path}")
    else:
        print(f"\nZnaleziono plik `{destination_file_path}`. Pomijam pobieranie...")


if __name__ == '__main__':
    download_dataset(
        file_url=DATASET_TID2008_URL,
        destination_file_path=DATASET_TID2008_ARCHIVE_PATH
    )

    download_dataset(
        file_url=DATASET_TID2013_URL,
        destination_file_path=DATASET_TID2013_ARCHIVE_PATH
    )

    download_dataset(
        file_url=DATASET_KADID10K_URL,
        destination_file_path=DATASET_KADID10K_ARCHIVE_PATH
    )
