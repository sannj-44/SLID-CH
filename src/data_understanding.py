import pickle
from pathlib import Path
import numpy as np


DATASET = Path(
    r"IMUSlip\IMUSlip\data\0-Trials"
)


def inspect_imu_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print("=" * 70)
    print("FILE:", file_path)
    print("=" * 70)

    print("Main object type:", type(data))

    if not isinstance(data, dict):
        print("Unexpected structure.")
        return

    print("\nIMU units:")
    for imu_name, imu_data in data.items():

        print(f"\n--- {imu_name} ---")
        print("Type:", type(imu_data))
        print("Channels:", list(imu_data.keys()))

        for channel, values in imu_data.items():

            values = np.asarray(values, dtype=float)

            print(
                f"{channel:>3} | "
                f"samples={len(values):6d} | "
                f"dtype={values.dtype} | "
                f"min={np.min(values):10.4f} | "
                f"max={np.max(values):10.4f} | "
                f"mean={np.mean(values):10.4f} | "
                f"std={np.std(values):10.4f} | "
                f"missing={np.isnan(values).sum():5d}"
            )


def main():

    files = list(DATASET.rglob("IMU_data.pkl"))

    print("Number of IMU trial files:", len(files))

    if not files:
        print("No IMU_data.pkl files found.")
        return

    # Inspect first trial
    inspect_imu_file(files[0])

    print("\nFirst trial:")
    print(files[0])

    print("\nExample trials:")
    for file in files[:10]:
        print(file)


if __name__ == "__main__":
    main()