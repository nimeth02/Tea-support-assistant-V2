"""Download the tea leaf disease dataset from Kaggle into dataset/data/."""

import argparse
import shutil
from pathlib import Path

import kagglehub

DATASET_HANDLE = "bmshahriaalam/tealeafbd-tea-leaf-disease-detection"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def download_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Download DATASET_HANDLE via kagglehub and copy it into output_dir."""
    cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in cache_path.iterdir():
        dest = output_dir / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the tea leaf disease dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to copy the dataset into (default: dataset/data/)",
    )
    args = parser.parse_args()

    output_dir = download_dataset(args.output_dir)
    print(f"Dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
