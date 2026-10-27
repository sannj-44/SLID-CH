from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SLID-CH: TEMPORAL FEATURE ENGINEERING
# ============================================================
# Input:
#   data_processed/slid_ch_features.csv
#
# Output:
#   data_processed/slid_ch_features_temporal.csv
#
# Safety:
#   - Original dataset is never modified.
#   - No temporal calculations cross trial boundaries.
#   - Label-derived columns are retained as metadata but are
#     excluded from the feature-generation process.
#   - The target and existing trial split are preserved.
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "slid_ch_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "slid_ch_features_temporal.csv"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "trial_split.json"
)


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

# Feature suffixes from the original 72 IMU statistical features.
IMU_FEATURE_SUFFIXES = (
    "_mean",
    "_std",
    "_min",
    "_max",
    "_range",
    "_rms",
)

# Existing metadata and label-related columns.
TARGET_COLUMN = "slip"
TRIAL_COLUMN = "trial"
OBJECT_COLUMN = "object"

TIME_START_COLUMN = "time_start"
TIME_END_COLUMN = "time_end"

LABEL_DERIVED_COLUMNS = [
    "max_contact_speed",
    "slip_fraction",
]


# ------------------------------------------------------------
# 3. LOAD AND VALIDATE THE DATASET
# ------------------------------------------------------------

def load_dataset():
    print("=" * 70)
    print("SLID-CH TEMPORAL FEATURE ENGINEERING")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Existing trial split not found:\n{SPLIT_FILE}"
        )

    print(f"\nInput dataset:\n{INPUT_FILE}")

    # Read the original file without modifying it.
    df = pd.read_csv(INPUT_FILE)

    print(f"\nOriginal rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    required_columns = [
        TRIAL_COLUMN,
        OBJECT_COLUMN,
        TIME_START_COLUMN,
        TIME_END_COLUMN,
        TARGET_COLUMN,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}"
        )

    if df[TRIAL_COLUMN].isna().any():
        raise ValueError(
            "Missing trial identifiers were found. "
            "Fix these before calculating temporal features."
        )

    if df[OBJECT_COLUMN].isna().any():
        raise ValueError(
            "Missing object identifiers were found. "
            "Fix these before calculating temporal features."
        )

    # A trial key must identify one independent time sequence.
    # Including object prevents accidental mixing if trial names
    # are reused under different objects.
    df["_temporal_group"] = (
        df[OBJECT_COLUMN].astype(str)
        + "::"
        + df[TRIAL_COLUMN].astype(str)
    )

    return df


# ------------------------------------------------------------
# 4. IDENTIFY THE ORIGINAL IMU FEATURES
# ------------------------------------------------------------

def identify_imu_features(df):
    imu_features = [
        col
        for col in df.columns
        if col.endswith(IMU_FEATURE_SUFFIXES)
    ]

    if not imu_features:
        raise ValueError(
            "No IMU statistical features were identified."
        )

    # Explicitly exclude the target and label-derived columns.
    forbidden = {
        TARGET_COLUMN,
        *LABEL_DERIVED_COLUMNS,
        TRIAL_COLUMN,
        OBJECT_COLUMN,
        TIME_START_COLUMN,
        TIME_END_COLUMN,
        "window_duration",
    }

    imu_features = [
        col for col in imu_features
        if col not in forbidden
    ]

    print(f"\nOriginal IMU features identified: {len(imu_features)}")

    if len(imu_features) != 72:
        print(
            "WARNING: Expected 72 IMU features, "
            f"but identified {len(imu_features)}."
        )

    print("\nExample original features:")
    for col in imu_features[:10]:
        print(f"  - {col}")

    return imu_features


# ------------------------------------------------------------
# 5. GENERATE TEMPORAL FEATURES
# ------------------------------------------------------------

def create_temporal_features(df, imu_features):
    print("\nCreating temporal features...")

    # Sort chronologically within each independent trial.
    df = df.sort_values(
        [
            "_temporal_group",
            TIME_START_COLUMN,
            TIME_END_COLUMN,
        ],
        kind="mergesort",
    ).copy()

    grouped = df.groupby(
        "_temporal_group",
        sort=False,
        observed=True,
    )

    generated_columns = []

    # --------------------------------------------------------
    # A. First-order differences
    # --------------------------------------------------------
    # Delta = current window value - previous window value.
    #
    # A positive delta means the feature increased.
    # A negative delta means the feature decreased.
    #
    # The first window of every trial has no previous window.
    # Its delta is set to zero.
    # --------------------------------------------------------

    for feature in imu_features:
        delta_name = f"{feature}_delta"

        df[delta_name] = (
            grouped[feature]
            .diff()
            .fillna(0.0)
            .astype(np.float32)
        )

        generated_columns.append(delta_name)

    print(
        f"Created {len(generated_columns)} first-order "
        "difference features."
    )

    # --------------------------------------------------------
    # B. Absolute first-order differences
    # --------------------------------------------------------
    # Measures the magnitude of change regardless of direction.
    # --------------------------------------------------------

    for feature in imu_features:
        abs_delta_name = f"{feature}_abs_delta"

        df[abs_delta_name] = (
            df[f"{feature}_delta"]
            .abs()
            .astype(np.float32)
        )

        generated_columns.append(abs_delta_name)

    print(
        "Created absolute-difference features."
    )

    # --------------------------------------------------------
    # C. Change rates
    # --------------------------------------------------------
    # Change rate = feature difference / elapsed time.
    #
    # The elapsed time is calculated between the current
    # window's start and the previous window's start.
    #
    # If elapsed time is zero or invalid, rate is set to zero.
    # --------------------------------------------------------

    previous_start = grouped[TIME_START_COLUMN].shift(1)

    elapsed_time = (
        df[TIME_START_COLUMN] - previous_start
    )

    valid_elapsed = (
        elapsed_time.notna()
        & np.isfinite(elapsed_time)
        & (elapsed_time > 0)
    )

    for feature in imu_features:
        rate_name = f"{feature}_change_rate"

        delta = df[f"{feature}_delta"]

        rate = pd.Series(
            0.0,
            index=df.index,
            dtype=np.float64,
        )

        rate.loc[valid_elapsed] = (
            delta.loc[valid_elapsed]
            / elapsed_time.loc[valid_elapsed]
        )

        # Avoid invalid infinite values.
        rate = rate.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)

        df[rate_name] = rate.astype(np.float32)

        generated_columns.append(rate_name)

    print("Created change-rate features.")

    # --------------------------------------------------------
    # D. Verify that generated features are valid
    # --------------------------------------------------------

    temporal_values = df[generated_columns].to_numpy()

    if not np.isfinite(temporal_values).all():
        raise ValueError(
            "Invalid values detected in temporal features."
        )

    print(
        f"\nTotal new temporal features: "
        f"{len(generated_columns)}"
    )

    return df, generated_columns


# ------------------------------------------------------------
# 6. VERIFY DATA INTEGRITY
# ------------------------------------------------------------

def validate_output(original_df, engineered_df, imu_features):
    print("\nValidating the engineered dataset...")

    if len(original_df) != len(engineered_df):
        raise AssertionError(
            "The row count changed unexpectedly."
        )

    # Verify that original labels remain unchanged.
    original_labels = (
        original_df.set_index(
            ["_temporal_group", TIME_START_COLUMN, TIME_END_COLUMN]
        )[TARGET_COLUMN]
        .sort_index()
    )

    engineered_labels = (
        engineered_df.set_index(
            ["_temporal_group", TIME_START_COLUMN, TIME_END_COLUMN]
        )[TARGET_COLUMN]
        .sort_index()
    )

    pd.testing.assert_series_equal(
        original_labels,
        engineered_labels,
        check_names=True,
    )

    # Verify that original IMU feature values remain unchanged.
    # Sorting is necessary because rows are chronologically ordered.
    original_values = (
        original_df.set_index(
            ["_temporal_group", TIME_START_COLUMN, TIME_END_COLUMN]
        )[imu_features]
        .sort_index()
    )

    engineered_values = (
        engineered_df.set_index(
            ["_temporal_group", TIME_START_COLUMN, TIME_END_COLUMN]
        )[imu_features]
        .sort_index()
    )

    pd.testing.assert_frame_equal(
        original_values,
        engineered_values,
        check_exact=True,
    )

    print("  PASS: Row count preserved.")
    print("  PASS: Original labels preserved.")
    print("  PASS: Original IMU feature values preserved.")
    print("  PASS: Temporal features contain finite values.")


# ------------------------------------------------------------
# 7. SAVE THE NEW DATASET
# ------------------------------------------------------------

def save_dataset(df):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only the temporary grouping column.
    # The original metadata columns remain in the output.
    output_df = df.drop(
        columns=["_temporal_group"]
    )

    # Write to a separate output file.
    # Never overwrite the original dataset.
    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8g",
    )

    print(f"\nNew dataset saved to:\n{OUTPUT_FILE}")

    print(f"\nOutput rows: {len(output_df):,}")
    print(f"Output columns: {len(output_df.columns)}")
    print(
        "Additional columns: "
        f"{len(output_df.columns) - 80}"
    )


# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------

def main():
    df = load_dataset()

    # Preserve an untouched in-memory copy for integrity checks.
    original_df = df.copy(deep=True)

    imu_features = identify_imu_features(df)

    engineered_df, generated_columns = (
        create_temporal_features(
            df,
            imu_features,
        )
    )

    validate_output(
        original_df,
        engineered_df,
        imu_features,
    )

    save_dataset(engineered_df)

    print("\n" + "=" * 70)
    print("TEMPORAL FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print("\nSummary:")
    print(f"  Original IMU features: {len(imu_features)}")
    print(f"  New temporal features: {len(generated_columns)}")
    print(f"  Final dataset rows: {len(engineered_df):,}")
    print(f"  Final dataset columns: {len(engineered_df.columns) - 1}")
    print("\nOriginal dataset remains unchanged.")
    print("No model was trained in this step.")


if __name__ == "__main__":
    main()