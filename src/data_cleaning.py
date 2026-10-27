import pickle
from pathlib import Path
import numpy as np


DATASET = Path(
    r"IMUSlip\IMUSlip\data\0-Trials"
)

EXPECTED_IMUS = {
    "Left IMU Signal",
    "Right IMU Signal"
}

EXPECTED_CHANNELS = {
    "ax", "ay", "az",
    "gx", "gy", "gz"
}


def check_trial(file_path):

    problems = []

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return [f"Could not load file: {e}"]

    # Check top-level structure
    if not isinstance(data, dict):
        return ["Top-level object is not a dictionary"]

    # Check IMU names
    actual_imus = set(data.keys())

    if actual_imus != EXPECTED_IMUS:
        problems.append(
            f"Unexpected IMUs: {actual_imus}"
        )

    all_lengths = []

    for imu_name in EXPECTED_IMUS:

        if imu_name not in data:
            problems.append(
                f"Missing IMU: {imu_name}"
            )
            continue

        imu = data[imu_name]

        if not isinstance(imu, dict):
            problems.append(
                f"{imu_name} is not a dictionary"
            )
            continue

        # Check channels
        actual_channels = set(imu.keys())

        missing_channels = EXPECTED_CHANNELS - actual_channels
        extra_channels = actual_channels - EXPECTED_CHANNELS

        if missing_channels:
            problems.append(
                f"{imu_name} missing channels: {missing_channels}"
            )

        if extra_channels:
            problems.append(
                f"{imu_name} extra channels: {extra_channels}"
            )

        for channel in EXPECTED_CHANNELS:

            if channel not in imu:
                continue

            values = np.asarray(
                imu[channel],
                dtype=float
            )

            # Empty channel
            if len(values) == 0:
                problems.append(
                    f"{imu_name}/{channel}: empty"
                )
                continue

            all_lengths.append(len(values))

            # NaN
            nan_count = np.isnan(values).sum()

            if nan_count > 0:
                problems.append(
                    f"{imu_name}/{channel}: "
                    f"{nan_count} NaN values"
                )

            # Infinite
            inf_count = np.isinf(values).sum()

            if inf_count > 0:
                problems.append(
                    f"{imu_name}/{channel}: "
                    f"{inf_count} infinite values"
                )

    # Check channel lengths
    if all_lengths and len(set(all_lengths)) != 1:
        problems.append(
            f"Unequal channel lengths: "
            f"{sorted(set(all_lengths))}"
        )

    return problems


def main():

    files = sorted(
        DATASET.rglob("IMU_data.pkl")
    )

    print("=" * 70)
    print("DATA CLEANING CHECK")
    print("=" * 70)

    print(f"Total IMU trials: {len(files)}")

    valid_trials = 0
    invalid_trials = 0

    for file_path in files:

        problems = check_trial(file_path)

        if problems:
            invalid_trials += 1

            print("\nPROBLEM:")
            print(file_path)

            for problem in problems:
                print("  -", problem)

        else:
            valid_trials += 1

    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)

    print("Total trials :", len(files))
    print("Valid trials :", valid_trials)
    print("Invalid trials:", invalid_trials)


if __name__ == "__main__":
    main()
    