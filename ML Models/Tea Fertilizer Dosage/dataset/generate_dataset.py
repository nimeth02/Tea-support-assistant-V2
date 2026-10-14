"""Generate the synthetic tea fertilizer dosage dataset into dataset/data/.

There is no real historical dataset of fertilizer applications and outcomes
anywhere in this repo, so this produces one from a documented, made-up
formula instead - see the module docstring on `generate_synthetic_dataset`
for the reasoning. It's a placeholder for the training *pipeline*, not a
source of real agronomic recommendations.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_N_SAMPLES = 800
DEFAULT_SEED = 42


def generate_synthetic_dataset(n_samples: int = DEFAULT_N_SAMPLES, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Make up a fertilizer-dosage dataset across the full range of inputs the app allows.

    The formula, in plain words:
    - Start from a base rate of fertilizer per acre for a standard ~90-day cycle.
    - Scale it up for bigger land (`landSize`) and for denser planting
      (`plantAmount` relative to land size - a crowded plot needs more product
      than a sparse one of the same size).
    - Scale it up the longer it's been since the last fertilization
      (`days_since_last_fert`) - a 180-day gap needs roughly double a 90-day gap.
    - Add a small bump for steeper slopes (`degree`), since steeper land loses
      more nutrients to runoff.
    - Add about 6% random noise per sample, since nothing in the real world is
      perfectly predictable from four numbers.
    """
    rng = np.random.default_rng(seed)

    land_size = rng.uniform(0.25, 6.0, n_samples)              # acres
    density = rng.uniform(6000, 14000, n_samples)              # plants per acre
    plant_amount = np.round(land_size * density).astype(int)   # total plants
    degree = rng.integers(0, 11, n_samples)                     # slope, 0-10
    days_since_last_fert = rng.uniform(14, 400, n_samples)      # days

    BASE_KG_PER_ACRE_PER_90_DAYS = 18.0   # made-up baseline rate
    BASELINE_DENSITY = 10000.0            # plants/acre this baseline assumes

    density_factor = density / BASELINE_DENSITY
    slope_factor = 1 + 0.025 * degree
    time_factor = days_since_last_fert / 90.0

    dosage = (
        land_size
        * BASE_KG_PER_ACRE_PER_90_DAYS
        * time_factor
        * density_factor
        * slope_factor
    )
    noise = rng.normal(0, 0.06, n_samples) * dosage  # ~6% relative noise
    dosage = np.clip(dosage + noise, 0.5, None)

    return pd.DataFrame({
        "landSize": land_size.round(2),
        "plantAmount": plant_amount,
        "degree": degree,
        "days_since_last_fert": days_since_last_fert.round(0).astype(int),
        "dosage_kg": dosage.round(2),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic tea fertilizer dosage dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Directory to write the dataset into (default: dataset/data/)")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    df = generate_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "tea_fertilizer_dosage.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset ready at: {output_path.resolve()} ({len(df)} rows)")


if __name__ == "__main__":
    main()
