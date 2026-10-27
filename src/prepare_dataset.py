from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    PROJECT_ROOT
    / "IMUSlip"
    / "IMUSlip"
    / "data"
    / "0-Trials"
)

OUTPUT_DIR = PROJECT_ROOT / "data_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "slid_ch_features.csv"
OUTPUT_PKL = OUTPUT_DIR / "slid_ch_features.pkl"

WINDOW_SIZE = 30

# Existing slip threshold. Verify against the dataset's
# official labeling definition before interpreting results.
SLIP_THRESHOLD = 0.06

IMU_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

IMUS = [
    "Left IMU Signal",
    "Right IMU Signal",
]

CONTACT_COMPONENTS = ["x", "y", "z"]


# ============================================================
# PICKLE LOADING
# ============================================================

def load_pickle(path):
    """Load a pickle file without modifying it."""

    with open(path, "rb") as file:
        return pickle.load(file)


# ============================================================
# SAFE NUMERIC ARRAY CONVERSION
# ============================================================

def safe_array(values, n_samples=None):
    """
    Convert numeric values, None, and invalid values to a
    float array.

    Missing or invalid values become NaN.

    If n_samples is specified, pad short arrays with NaN
    or truncate long arrays to the required length.
    """

    result = []

    for value in values:
        try:
            if value is None:
                result.append(np.nan)
            else:
                number = float(value)

                if np.isfinite(number):
                    result.append(number)
                else:
                    result.append(np.nan)

        except (ValueError, TypeError, OverflowError):
            result.append(np.nan)

    array = np.asarray(result, dtype=np.float64)

    if n_samples is not None:

        if len(array) < n_samples:
            array = np.pad(
                array,
                (0, n_samples - len(array)),
                constant_values=np.nan,
            )

        elif len(array) > n_samples:
            array = array[:n_samples]

    return array


# ============================================================
# CONTACT VELOCITY
# ============================================================

def vector_speed(velocity_dict, n_samples):
    """
    Calculate 3D contact speed:

        speed = sqrt(vx^2 + vy^2 + vz^2)

    Missing or invalid vector components produce NaN.
    """

    components = {}

    for component in CONTACT_COMPONENTS:

        if component not in velocity_dict:
            return np.full(n_samples, np.nan)

        components[component] = safe_array(
            velocity_dict[component],
            n_samples=n_samples,
        )

    x = components["x"]
    y = components["y"]
    z = components["z"]

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    speed = np.full(n_samples, np.nan)

    speed[valid] = np.sqrt(
        x[valid] ** 2
        + y[valid] ** 2
        + z[valid] ** 2
    )

    return speed


# ============================================================
# SLIP LABEL GENERATION
# ============================================================

def create_slip_labels(velocity_data, n_samples):
    """
    Create sample-level slip labels.

    Label:
        1 = at least one valid contact speed exceeds threshold
        0 = valid contact speeds exist, but none exceed threshold
       -1 = no valid contact velocity is available

    This preserves the existing threshold-based labeling rule.
    """

    contact_speeds = []

    for contact_name, velocity in velocity_data.items():

        if not isinstance(velocity, dict):
            continue

        if not all(
            component in velocity
            for component in CONTACT_COMPONENTS
        ):
            continue

        speed = vector_speed(
            velocity,
            n_samples,
        )

        contact_speeds.append(speed)

    if not contact_speeds:
        raise ValueError(
            "No valid contact velocity data found."
        )

    contact_speeds = np.vstack(contact_speeds)

    # Identify samples for which every contact speed is missing.
    no_velocity = np.all(
        ~np.isfinite(contact_speeds),
        axis=0,
    )

    # Replace missing values with -infinity temporarily.
    # This avoids calling nanmax on all-NaN columns.
    valid_speeds = np.where(
        np.isfinite(contact_speeds),
        contact_speeds,
        -np.inf,
    )

    max_speed = np.max(
        valid_speeds,
        axis=0,
    )

    # Restore missing values for samples with no valid contacts.
    max_speed[no_velocity] = np.nan

    # Default every label to unavailable.
    labels = np.full(
        n_samples,
        -1,
        dtype=np.int8,
    )

    valid_velocity = ~no_velocity

    labels[valid_velocity] = (
        max_speed[valid_velocity] > SLIP_THRESHOLD
    ).astype(np.int8)

    return labels, max_speed


# ============================================================
# STATISTICAL FEATURE EXTRACTION
# ============================================================

def extract_window_features(window):
    """
    Extract six statistical features from each of the
    12 IMU channels:

        Mean
        Standard deviation
        Minimum
        Maximum
        Range
        Root mean square
    """

    features = {}

    for imu_name, channels in window.items():

        imu_short = (
            "left"
            if imu_name.startswith("Left")
            else "right"
        )

        for channel_name in IMU_CHANNELS:

            values = np.asarray(
                channels[channel_name],
                dtype=np.float64,
            )

            prefix = f"{imu_short}_{channel_name}"

            features[f"{prefix}_mean"] = float(
                np.mean(values)
            )

            features[f"{prefix}_std"] = float(
                np.std(values)
            )

            features[f"{prefix}_min"] = float(
                np.min(values)
            )

            features[f"{prefix}_max"] = float(
                np.max(values)
            )

            features[f"{prefix}_range"] = float(
                np.ptp(values)
            )

            features[f"{prefix}_rms"] = float(
                np.sqrt(np.mean(values ** 2))
            )

    return features


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(trial_dir):

    imu_file = trial_dir / "IMU_data.pkl"

    velocity_file = (
        trial_dir
        / "smoothed_contact_sliding_velocities.pkl"
    )

    time_file = trial_dir / "times.pkl"

    if not imu_file.exists():
        return []

    if not velocity_file.exists():
        return []

    if not time_file.exists():
        return []

    imu_data = load_pickle(imu_file)
    velocity_data = load_pickle(velocity_file)
    time_data = load_pickle(time_file)

    # --------------------------------------------------------
    # Validate IMU structure
    # --------------------------------------------------------

    if not isinstance(imu_data, dict):
        raise ValueError(
            f"Invalid IMU structure in {trial_dir}"
        )

    for imu_name in IMUS:

        if imu_name not in imu_data:
            raise ValueError(
                f"Missing {imu_name} in {trial_dir}"
            )

        for channel in IMU_CHANNELS:

            if channel not in imu_data[imu_name]:
                raise ValueError(
                    f"Missing channel {channel} in {trial_dir}"
                )

    left = imu_data["Left IMU Signal"]

    n_samples = len(left["ax"])

    if n_samples < WINDOW_SIZE:
        return []

    # --------------------------------------------------------
    # Convert IMU channels to arrays
    # --------------------------------------------------------

    imu_arrays = {}

    for imu_name in IMUS:

        imu_arrays[imu_name] = {}

        for channel in IMU_CHANNELS:

            values = safe_array(
                imu_data[imu_name][channel]
            )

            if len(values) != n_samples:
                raise ValueError(
                    f"Channel length mismatch: "
                    f"{imu_name}/{channel} in {trial_dir}"
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid IMU values in "
                    f"{imu_name}/{channel}, {trial_dir}"
                )

            imu_arrays[imu_name][channel] = values

    # --------------------------------------------------------
    # Create sample-level labels
    # --------------------------------------------------------

    labels, max_speed = create_slip_labels(
        velocity_data,
        n_samples,
    )

    # --------------------------------------------------------
    # Load and validate timestamps
    # --------------------------------------------------------

    if not isinstance(time_data, dict) or "time" not in time_data:
        raise ValueError(
            f"Invalid timestamp structure in {trial_dir}"
        )

    times = safe_array(
        time_data["time"]
    )

    if len(times) != n_samples:
        raise ValueError(
            f"Timestamp/IMU length mismatch in {trial_dir}"
        )

    if not np.isfinite(times).all():
        raise ValueError(
            f"Invalid timestamps in {trial_dir}"
        )

    # --------------------------------------------------------
    # Trial metadata
    # --------------------------------------------------------

    object_name = trial_dir.parent.name
    trial_name = trial_dir.name

    rows = []

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for end_idx in range(
        WINDOW_SIZE,
        n_samples + 1,
    ):

        start_idx = end_idx - WINDOW_SIZE

        window_labels = labels[
            start_idx:end_idx
        ]

        # Skip windows where every label is unavailable.
        valid_labels = window_labels[
            window_labels >= 0
        ]

        if len(valid_labels) == 0:
            continue

        # ----------------------------------------------------
        # Window target
        # ----------------------------------------------------

        slip_fraction = float(
            np.mean(valid_labels)
        )

        label = int(
            slip_fraction >= 0.5
        )

        # ----------------------------------------------------
        # Construct IMU window
        # ----------------------------------------------------

        window = {}

        for imu_name in IMUS:

            window[imu_name] = {}

            for channel in IMU_CHANNELS:

                window[imu_name][channel] = (
                    imu_arrays[imu_name][channel]
                    [start_idx:end_idx]
                )

        # ----------------------------------------------------
        # Extract IMU features
        # ----------------------------------------------------

        features = extract_window_features(
            window
        )

        # ----------------------------------------------------
        # Additional information
        # ----------------------------------------------------

        features["time_start"] = float(
            times[start_idx]
        )

        features["time_end"] = float(
            times[end_idx - 1]
        )

        features["window_duration"] = float(
            times[end_idx - 1] - times[start_idx]
        )

        # Contact velocity is retained for analysis only.
        # Do not use it as an input to the ML model.
        window_speeds = max_speed[
            start_idx:end_idx
        ]

        valid_speeds = window_speeds[
            np.isfinite(window_speeds)
        ]

        features["max_contact_speed"] = (
            float(np.max(valid_speeds))
            if len(valid_speeds) > 0
            else np.nan
        )

        # Label-derived information: analysis only.
        # Exclude this column from model inputs.
        features["slip_fraction"] = slip_fraction

        # Metadata
        features["object"] = object_name
        features["trial"] = trial_name

        # Target
        features["slip"] = label

        rows.append(features)

    return rows


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    print("=" * 80)
    print("SLID-CH DATA PREPROCESSING")
    print("=" * 80)

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET}"
        )

    imu_files = sorted(
        DATASET.rglob("IMU_data.pkl")
    )

    print(
        f"\nRaw trials found: {len(imu_files)}"
    )

    if not imu_files:
        raise RuntimeError(
            "No IMU_data.pkl files were found."
        )

    all_rows = []

    processed_trials = 0
    failed_trials = 0
    skipped_trials = 0

    # --------------------------------------------------------
    # Process all trials
    # --------------------------------------------------------

    for i, imu_file in enumerate(
        imu_files,
        start=1,
    ):

        trial_dir = imu_file.parent

        try:

            rows = process_trial(
                trial_dir
            )

            if not rows:
                skipped_trials += 1

                print(
                    f"[{i:03d}/{len(imu_files)}] "
                    f"Skipped: {trial_dir.name} "
                    f"(no usable windows)"
                )

                continue

            all_rows.extend(rows)

            processed_trials += 1

            print(
                f"[{i:03d}/{len(imu_files)}] "
                f"Processed: {trial_dir.name} "
                f"-> {len(rows)} windows"
            )

        except Exception as error:

            failed_trials += 1

            print(
                f"[{i:03d}/{len(imu_files)}] "
                f"FAILED: {trial_dir}"
            )

            print(
                f"       Reason: {error}"
            )

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------

    if not all_rows:
        raise RuntimeError(
            "No usable windows were generated."
        )

    df = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # Validate generated features
    # --------------------------------------------------------

    imu_feature_columns = [
        column
        for column in df.columns
        if column.endswith(
            (
                "_mean",
                "_std",
                "_min",
                "_max",
                "_range",
                "_rms",
            )
        )
    ]

    if not imu_feature_columns:
        raise RuntimeError(
            "No IMU feature columns were generated."
        )

    if not np.isfinite(
        df[imu_feature_columns].to_numpy(
            dtype=np.float64
        )
    ).all():

        raise ValueError(
            "Generated IMU features contain NaN or infinity."
        )

    if not df["slip"].isin([0, 1]).all():
        raise ValueError(
            "Unexpected target labels were generated."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(OUTPUT_PKL, "wb") as file:
        pickle.dump(
            df,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"Trials found       : {len(imu_files)}"
    )

    print(
        f"Trials processed   : {processed_trials}"
    )

    print(
        f"Trials skipped     : {skipped_trials}"
    )

    print(
        f"Trials failed      : {failed_trials}"
    )

    print(
        f"\nTotal windows      : {len(df):,}"
    )

    print(
        f"Total columns      : {len(df.columns)}"
    )

    print(
        f"IMU feature columns: {len(imu_feature_columns)}"
    )

    print("\nClass distribution:")

    print(
        df["slip"]
        .value_counts()
        .sort_index()
    )

    print("\nClass percentage:")

    print(
        (
            df["slip"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    print("\nObjects:")

    print(
        df["object"]
        .value_counts()
    )

    print("\nOutput files:")

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {OUTPUT_PKL}"
    )

    print("\nFirst five rows:")

    print(
        df.head().to_string(index=False)
    )

    print("\nOriginal dataset files were not modified.")


if __name__ == "__main__":
    main()