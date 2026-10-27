from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRIALS_ROOT = (
    PROJECT_ROOT
    / "IMUSlip"
    / "IMUSlip"
    / "data"
    / "0-Trials"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "original"
    / "raw_imu_data.csv"
)

IMU_KEYS = {
    "Left IMU Signal": "left",
    "Right IMU Signal": "right"
}

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

SIGNAL_COLUMNS = [
    f"{side}_{channel}"
    for side in ("left", "right")
    for channel in CHANNELS
]


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def main():

    if not TRIALS_ROOT.exists():
        sys.exit(
            f"ERROR: Raw trial directory not found:\n{TRIALS_ROOT}"
        )

    imu_files = sorted(TRIALS_ROOT.rglob("IMU_data.pkl"))

    if not imu_files:
        sys.exit("ERROR: No IMU_data.pkl files found.")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_rows = 0
    failures = []
    first_write = True

    print(f"Trials found: {len(imu_files)}")
    print("Exporting original IMU data to CSV...")

    for number, imu_path in enumerate(imu_files, start=1):

        try:
            imu_data = load_pickle(imu_path)

            time_path = imu_path.parent / "times.pkl"

            if not isinstance(imu_data, dict):
                raise ValueError("IMU data is not a dictionary.")

            signals = {}
            lengths = []

            for imu_key, side in IMU_KEYS.items():

                if (
                    imu_key not in imu_data
                    or not isinstance(imu_data[imu_key], dict)
                ):
                    raise ValueError(
                        f"Missing or invalid IMU: {imu_key}"
                    )

                for channel in CHANNELS:

                    if channel not in imu_data[imu_key]:
                        raise ValueError(
                            f"Missing channel: {channel}"
                        )

                    values = np.asarray(
                        imu_data[imu_key][channel],
                        dtype=float
                    ).reshape(-1)

                    signals[f"{side}_{channel}"] = values
                    lengths.append(len(values))

            if len(set(lengths)) != 1:
                raise ValueError(
                    f"IMU channel lengths differ: {set(lengths)}"
                )

            if not time_path.exists():
                raise ValueError("times.pkl is missing.")

            time_data = load_pickle(time_path)

            if isinstance(time_data, dict):
                if "time" not in time_data:
                    raise ValueError(
                        "Timestamp dictionary has no 'time' key."
                    )

                timestamps = np.asarray(
                    time_data["time"],
                    dtype=float
                ).reshape(-1)

            else:
                timestamps = np.asarray(
                    time_data,
                    dtype=float
                ).reshape(-1)

            if len(timestamps) != lengths[0]:
                raise ValueError(
                    "Timestamp and IMU sample counts differ."
                )

            frame = pd.DataFrame(signals)

            frame.insert(0, "time", timestamps)
            frame.insert(0, "trial", imu_path.parent.name)
            frame.insert(0, "object", imu_path.parent.parent.name)

            frame.to_csv(
                OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )

            first_write = False
            total_rows += len(frame)

            if number % 25 == 0 or number == len(imu_files):
                print(
                    f"Processed {number}/{len(imu_files)} trials; "
                    f"{total_rows:,} rows"
                )

        except Exception as error:
            failures.append((imu_path, str(error)))

    if failures:
        print("\nERROR: Some trials could not be exported.")

        for path, reason in failures:
            print(f"\nFile: {path}")
            print(f"Reason: {reason}")

        sys.exit(
            "Export incomplete. Review the failures before proceeding."
        )

    print("\nEXPORT SUCCESSFUL")
    print(f"Trials exported: {len(imu_files)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()