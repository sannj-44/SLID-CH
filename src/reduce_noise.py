from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data_processed"

ORIGINAL_FILE = (
    DATA_DIR
    / "original"
    / "raw_imu_data.csv"
)

NOISY_FILE = (
    DATA_DIR
    / "noisy"
    / "noisy_imu_data.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "noise_reduced"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "filtered_imu_data.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "noise_reduction_metrics.csv"
)

FILTER_WINDOW = 3

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
    print("SLID-CH: MOVING-AVERAGE NOISE REDUCTION")
    print("=" * 65)

    if not ORIGINAL_FILE.exists():
        raise FileNotFoundError(
            f"Original CSV not found:\n{ORIGINAL_FILE}"
        )

    if not NOISY_FILE.exists():
        raise FileNotFoundError(
            f"Noisy CSV not found:\n{NOISY_FILE}\n"
            "Run src/add_noise.py first."
        )

    original = pd.read_csv(ORIGINAL_FILE)
    noisy = pd.read_csv(NOISY_FILE)

    required_columns = (
        GROUP_COLUMNS + ["time"] + IMU_CHANNELS
    )

    for name, dataframe in [
        ("Original", original),
        ("Noisy", noisy),
    ]:

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"{name} CSV is missing columns: {missing}"
            )

        if not np.isfinite(
            dataframe[IMU_CHANNELS].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name} IMU data contains NaN or infinity."
            )

    if len(original) != len(noisy):
        raise ValueError(
            "Original and noisy CSV row counts differ."
        )

    # Generate sample indices in memory.
    for dataframe in [original, noisy]:

        if "sample_index" not in dataframe.columns:

            dataframe["sample_index"] = (
                dataframe.groupby(
                    GROUP_COLUMNS,
                    sort=False,
                ).cumcount()
            )

    # Validate trial and object alignment.
    for column in GROUP_COLUMNS:

        if not original[column].equals(noisy[column]):
            raise ValueError(
                f"Alignment mismatch in {column}."
            )

    # Validate timestamp alignment.
    if not np.allclose(
        original["time"].to_numpy(dtype=float),
        noisy["time"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError(
            "Original and noisy timestamps do not match."
        )

    # Validate sample ordering.
    if not np.array_equal(
        original["sample_index"].to_numpy(),
        noisy["sample_index"].to_numpy(),
    ):
        raise ValueError(
            "Sample ordering does not match."
        )

    filtered = noisy.copy()

    # Filter each trial independently.
    for _, indices in noisy.groupby(
        GROUP_COLUMNS,
        sort=False,
    ).groups.items():

        for channel in IMU_CHANNELS:

            signal = noisy.loc[
                indices, channel
            ]

            filtered.loc[
                indices, channel
            ] = (
                signal.rolling(
                    window=FILTER_WINDOW,
                    center=True,
                    min_periods=1,
                )
                .mean()
                .to_numpy()
            )

    # Calculate errors against original recordings.
    metrics = []

    for channel in IMU_CHANNELS:

        reference = original[channel].to_numpy(
            dtype=np.float64
        )

        noisy_values = noisy[channel].to_numpy(
            dtype=np.float64
        )

        filtered_values = filtered[channel].to_numpy(
            dtype=np.float64
        )

        noisy_error = noisy_values - reference
        filtered_error = filtered_values - reference

        noisy_mse = float(
            np.mean(noisy_error ** 2)
        )

        filtered_mse = float(
            np.mean(filtered_error ** 2)
        )

        noisy_mae = float(
            np.mean(np.abs(noisy_error))
        )

        filtered_mae = float(
            np.mean(np.abs(filtered_error))
        )

        mse_reduction = (
            (noisy_mse - filtered_mse)
            / noisy_mse
            * 100.0
            if noisy_mse > 0
            else 0.0
        )

        metrics.append({
            "channel": channel,
            "noisy_MSE": noisy_mse,
            "filtered_MSE": filtered_mse,
            "noisy_MAE": noisy_mae,
            "filtered_MAE": filtered_mae,
            "MSE_reduction_percent": mse_reduction,
        })

    metrics_df = pd.DataFrame(metrics)

    # Aggregate errors over all channels and samples.
    original_values = original[
        IMU_CHANNELS
    ].to_numpy(dtype=np.float64)

    noisy_values = noisy[
        IMU_CHANNELS
    ].to_numpy(dtype=np.float64)

    filtered_values = filtered[
        IMU_CHANNELS
    ].to_numpy(dtype=np.float64)

    noisy_error = noisy_values - original_values
    filtered_error = filtered_values - original_values

    noisy_mse = float(
        np.mean(noisy_error ** 2)
    )

    filtered_mse = float(
        np.mean(filtered_error ** 2)
    )

    noisy_rmse = float(np.sqrt(noisy_mse))
    filtered_rmse = float(np.sqrt(filtered_mse))

    noisy_mae = float(
        np.mean(np.abs(noisy_error))
    )

    filtered_mae = float(
        np.mean(np.abs(filtered_error))
    )

    reduction_percent = (
        (noisy_mse - filtered_mse)
        / noisy_mse
        * 100.0
        if noisy_mse > 0
        else 0.0
    )

    # Save output files.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    # Display final results.
    print(f"Trials processed: {original['trial'].nunique()}")
    print(f"Rows processed: {len(original):,}")
    print(f"Channels filtered: {len(IMU_CHANNELS)}")
    print(f"Filter window: {FILTER_WINDOW} samples")

    print("\nOVERALL RESULTS")
    print("-" * 45)

    print(f"Noisy MSE: {noisy_mse:.8f}")
    print(f"Filtered MSE: {filtered_mse:.8f}")

    print(f"Noisy RMSE: {noisy_rmse:.8f}")
    print(f"Filtered RMSE: {filtered_rmse:.8f}")

    print(f"Noisy MAE: {noisy_mae:.8f}")
    print(f"Filtered MAE: {filtered_mae:.8f}")

    print(
        f"MSE reduction: {reduction_percent:.2f}%"
    )

    print("\nOutput files:")
    print(OUTPUT_FILE)
    print(METRICS_FILE)

    print("\nOriginal and noisy CSV files were not modified.")

    print(
        "\nNote: Metrics measure similarity to the original "
        "recordings, not necessarily real-world sensor-noise removal."
    )


if __name__ == "__main__":
    main()