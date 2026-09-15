from pathlib import Path
import pickle
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET = Path(r"IMUSlip\IMUSlip\data\0-Trials")

OUTPUT_DIR = Path("data_processed")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "slid_ch_features.csv"
OUTPUT_PKL = OUTPUT_DIR / "slid_ch_features.pkl"


# ============================================================
# PARAMETERS
# ============================================================

WINDOW_SIZE = 30

# Slip criterion from the SLID-CH dataset paper
SLIP_THRESHOLD = 0.06


# ============================================================
# IMU CHANNELS
# ============================================================

IMU_CHANNELS = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]

IMUS = [
    "Left IMU Signal",
    "Right IMU Signal",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_pickle(path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_array(values):
    """
    Convert a list containing numbers / None / NaN
    into a float numpy array.
    """
    result = []

    for value in values:
        if value is None:
            result.append(np.nan)
        else:
            try:
                result.append(float(value))
            except (ValueError, TypeError):
                result.append(np.nan)

    return np.asarray(result, dtype=float)


def vector_speed(velocity_dict, n_samples):
    """
    Calculate magnitude of a 3-D velocity vector.

    speed = sqrt(vx^2 + vy^2 + vz^2)

    Invalid / unavailable values become NaN.
    """

    x = safe_array(velocity_dict["x"])
    y = safe_array(velocity_dict["y"])
    z = safe_array(velocity_dict["z"])

    # Make sure all vectors have the expected length
    x = x[:n_samples]
    y = y[:n_samples]
    z = z[:n_samples]

    speed = np.sqrt(x**2 + y**2 + z**2)

    return speed


def create_slip_labels(velocity_data, n_samples):
    """
    Create sample-level slip labels.

    A sample is labelled as slip if ANY available contact
    point has speed > 0.06 m/s.

    Contacts:
        LP, LD1, LD2,
        RP, RD1, RD2
    """

    contact_speeds = []

    for contact_name, velocity in velocity_data.items():

        if not isinstance(velocity, dict):
            continue

        if not all(k in velocity for k in ["x", "y", "z"]):
            continue

        speed = vector_speed(velocity, n_samples)

        contact_speeds.append(speed)

    if not contact_speeds:
        raise ValueError("No valid contact velocity data found.")

    contact_speeds = np.vstack(contact_speeds)

    # Maximum available contact speed at every timestamp
    max_speed = np.nanmax(contact_speeds, axis=0)

    # Slip = speed above threshold
    labels = (max_speed > SLIP_THRESHOLD).astype(int)

    # Where absolutely no contact velocity exists
    no_velocity = np.all(np.isnan(contact_speeds), axis=0)

    labels[no_velocity] = -1

    return labels, max_speed


def extract_window_features(window):
    """
    Extract classical statistical features from a 30-sample window.

    For every IMU channel:

        mean
        std
        min
        max
        range
        RMS

    This converts the time-series window into fixed-length
    features suitable for classical ML.
    """

    features = {}

    for imu_name, channels in window.items():

        imu_short = "left" if imu_name.startswith("Left") else "right"

        for channel_name in IMU_CHANNELS:

            values = np.asarray(
                channels[channel_name],
                dtype=float
            )

            prefix = f"{imu_short}_{channel_name}"

            features[f"{prefix}_mean"] = np.mean(values)
            features[f"{prefix}_std"] = np.std(values)
            features[f"{prefix}_min"] = np.min(values)
            features[f"{prefix}_max"] = np.max(values)
            features[f"{prefix}_range"] = np.ptp(values)

            # Root Mean Square
            features[f"{prefix}_rms"] = np.sqrt(
                np.mean(values ** 2)
            )

    return features


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(trial_dir):

    imu_file = trial_dir / "IMU_data.pkl"
    velocity_file = trial_dir / "smoothed_contact_sliding_velocities.pkl"
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
    # Determine number of samples
    # --------------------------------------------------------

    left = imu_data["Left IMU Signal"]
    right = imu_data["Right IMU Signal"]

    n_samples = len(left["ax"])

    # --------------------------------------------------------
    # Check all IMU channels have same length
    # --------------------------------------------------------

    for imu_name in IMUS:

        for channel in IMU_CHANNELS:

            if len(imu_data[imu_name][channel]) != n_samples:
                raise ValueError(
                    f"Channel length mismatch in {trial_dir}"
                )

    # --------------------------------------------------------
    # Create sample-level labels
    # --------------------------------------------------------

    labels, max_speed = create_slip_labels(
        velocity_data,
        n_samples
    )

    times = np.asarray(
        time_data["time"],
        dtype=float
    )

    if len(times) != n_samples:
        raise ValueError(
            f"Time/IMU length mismatch in {trial_dir}"
        )

    # --------------------------------------------------------
    # Convert IMU data to numpy arrays
    # --------------------------------------------------------

    imu_arrays = {}

    for imu_name in IMUS:

        imu_arrays[imu_name] = {}

        for channel in IMU_CHANNELS:

            imu_arrays[imu_name][channel] = np.asarray(
                imu_data[imu_name][channel],
                dtype=float
            )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    object_name = trial_dir.parent.name
    trial_name = trial_dir.name

    rows = []

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for end_idx in range(WINDOW_SIZE, n_samples + 1):

        start_idx = end_idx - WINDOW_SIZE

        window_label_values = labels[start_idx:end_idx]

        # Skip windows where velocity information is completely
        # unavailable.
        if np.all(window_label_values == -1):
            continue

        # Remove unavailable label samples
        valid_labels = window_label_values[
            window_label_values >= 0
        ]

        if len(valid_labels) == 0:
            continue

        # Majority label inside the 30-sample window
        slip_fraction = np.mean(valid_labels)

        label = int(slip_fraction >= 0.5)

        # ----------------------------------------------------
        # Create IMU window
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
        # Extract statistical features
        # ----------------------------------------------------

        features = extract_window_features(window)

        # ----------------------------------------------------
        # Additional window-level information
        # ----------------------------------------------------

        features["time_start"] = times[start_idx]
        features["time_end"] = times[end_idx - 1]
        features["window_duration"] = (
            times[end_idx - 1] - times[start_idx]
        )

        features["max_contact_speed"] = np.nanmax(
            max_speed[start_idx:end_idx]
        )

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

    # Find all trials using IMU_data.pkl
    imu_files = sorted(
        DATASET.rglob("IMU_data.pkl")
    )

    print(f"\nRaw trials found: {len(imu_files)}")

    all_rows = []

    processed_trials = 0
    failed_trials = 0

    # --------------------------------------------------------
    # Process every trial
    # --------------------------------------------------------

    for i, imu_file in enumerate(imu_files, start=1):

        trial_dir = imu_file.parent

        try:

            rows = process_trial(trial_dir)

            all_rows.extend(rows)

            processed_trials += 1

            print(
                f"[{i:03d}/{len(imu_files)}] "
                f"Processed: {trial_dir.name} "
                f"-> {len(rows)} windows"
            )

        except Exception as e:

            failed_trials += 1

            print(
                f"[{i:03d}/{len(imu_files)}] "
                f"FAILED: {trial_dir}"
            )

            print(f"       Reason: {e}")

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise RuntimeError(
            "No samples were generated."
        )

    # --------------------------------------------------------
    # Check numerical columns
    # --------------------------------------------------------

    feature_columns = [
        c for c in df.columns
        if c not in [
            "object",
            "trial",
            "slip"
        ]
    ]

    # Check missing values
    missing_count = df[feature_columns].isna().sum().sum()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(df, f)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(f"Trials found       : {len(imu_files)}")
    print(f"Trials processed   : {processed_trials}")
    print(f"Trials failed      : {failed_trials}")

    print(f"\nTotal windows      : {len(df)}")
    print(f"Total features     : {len(df.columns)}")

    print(
        f"Numerical missing values : {missing_count}"
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
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_PKL}")

    print("\nFirst 5 rows:")
    print(
        df.head().to_string()
    )


if __name__ == "__main__":
    main()