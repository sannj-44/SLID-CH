from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "original"
    / "raw_imu_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "noisy"
)

OUTPUT_FILE = OUTPUT_DIR / "noisy_imu_data.csv"

NOISE_PERCENT = 50
NOISE_FRACTION = NOISE_PERCENT / 100.0

RANDOM_SEED = 42

GROUP_COLUMNS = ["object", "trial"]

IMU_CHANNELS = [
    "left_ax",
    "left_ay",
    "left_az",
    "left_gx",
    "left_gy",
    "left_gz",
    "right_ax",
    "right_ay",
    "right_az",
    "right_gx",
    "right_gy",
    "right_gz",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SLID-CH: GAUSSIAN NOISE ADDITION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Original CSV not found:\n{INPUT_FILE}"
        )

    original = pd.read_csv(INPUT_FILE)

    required_columns = (
        GROUP_COLUMNS + ["time"] + IMU_CHANNELS
    )

    missing = [
        column
        for column in required_columns
        if column not in original.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if not np.isfinite(
        original[IMU_CHANNELS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Original IMU data contains NaN or infinity."
        )

    # Generate sample indices in memory if absent.
    if "sample_index" not in original.columns:
        original["sample_index"] = (
            original.groupby(
                GROUP_COLUMNS,
                sort=False,
            ).cumcount()
        )

    noisy = original.copy()

    rng = np.random.default_rng(RANDOM_SEED)

    total_noise_squared = 0.0
    total_noise_absolute = 0.0
    total_values = 0

    # Generate independent noise for every trial/channel.
    for _, indices in original.groupby(
        GROUP_COLUMNS,
        sort=False,
    ).groups.items():

        for channel in IMU_CHANNELS:

            signal = original.loc[
                indices, channel
            ].to_numpy(dtype=np.float64)

            signal_sd = np.std(
                signal,
                ddof=0,
            )

            noise_sd = NOISE_FRACTION * signal_sd

            noise = rng.normal(
                loc=0.0,
                scale=noise_sd,
                size=len(signal),
            )

            noisy.loc[
                indices, channel
            ] = signal + noise

            total_noise_squared += float(
                np.sum(noise ** 2)
            )

            total_noise_absolute += float(
                np.sum(np.abs(noise))
            )

            total_values += len(noise)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    noisy.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    noise_rmse = np.sqrt(
        total_noise_squared / total_values
    )

    noise_mae = (
        total_noise_absolute / total_values
    )

    print(f"Trials processed: {original['trial'].nunique()}")
    print(f"Rows processed: {len(original):,}")
    print(f"IMU channels: {len(IMU_CHANNELS)}")
    print(f"Noise strength: {NOISE_PERCENT}% of signal SD")
    print(f"Random seed: {RANDOM_SEED}")

    print(f"\nAdded noise RMSE: {noise_rmse:.8f}")
    print(f"Added noise MAE: {noise_mae:.8f}")

    print(f"\nOutput file:\n{OUTPUT_FILE}")

    print("\nOriginal CSV was not modified.")
    print(
        "Note: This is simulated Gaussian noise, "
        "not measured physical sensor noise."
    )


if __name__ == "__main__":
    main()