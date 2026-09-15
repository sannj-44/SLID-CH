from pathlib import Path
import pickle
import numpy as np

DATASET = Path(r"IMUSlip\IMUSlip\data\0-Trials")

# Find first raw trial
imu_file = next(DATASET.rglob("IMU_data.pkl"))
trial_dir = imu_file.parent

print("=" * 80)
print("LABEL / TIMESTAMP STRUCTURE INSPECTION")
print("=" * 80)
print("Trial:", trial_dir)

FILES = [
    "times.pkl",
    "contact_sliding_velocities.pkl",
    "smoothed_contact_sliding_velocities.pkl",
]


def inspect_object(obj, name, indent=""):
    print(f"\n{name}")
    print("-" * 60)
    print("Type:", type(obj))

    if isinstance(obj, dict):
        print("Keys:", list(obj.keys()))

        for key, value in obj.items():
            print(f"\n{indent}{key}:")
            print(f"{indent}  Type: {type(value)}")

            if isinstance(value, dict):
                print(f"{indent}  Keys: {list(value.keys())}")

                for k2, v2 in value.items():
                    if v2 is None:
                        print(f"{indent}    {k2}: None")
                        continue

                    try:
                        arr = np.asarray(v2)
                        print(
                            f"{indent}    {k2}: "
                            f"shape={arr.shape}, "
                            f"dtype={arr.dtype}, "
                            f"size={arr.size}"
                        )

                        numeric = arr[
                            np.array(
                                [x is not None for x in arr.flatten()],
                                dtype=bool
                            )
                        ]

                        if numeric.size > 0:
                            try:
                                numeric = numeric.astype(float)
                                finite = numeric[np.isfinite(numeric)]

                                if finite.size > 0:
                                    print(
                                        f"{indent}      min={finite.min():.6f}, "
                                        f"max={finite.max():.6f}, "
                                        f"mean={finite.mean():.6f}"
                                    )
                                    print(
                                        f"{indent}      first={finite[:10]}"
                                    )
                            except (ValueError, TypeError):
                                print(
                                    f"{indent}      Non-numeric data"
                                )

                    except Exception as e:
                        print(f"{indent}    Could not inspect: {e}")

            else:
                try:
                    arr = np.asarray(value)

                    print(
                        f"{indent}  shape={arr.shape}, "
                        f"dtype={arr.dtype}, "
                        f"size={arr.size}"
                    )

                    flat = arr.flatten()

                    # Count None values
                    none_count = sum(x is None for x in flat)
                    print(f"{indent}  None count={none_count}")

                    # Try numeric conversion
                    valid = []

                    for x in flat:
                        if x is not None:
                            try:
                                fx = float(x)
                                if np.isfinite(fx):
                                    valid.append(fx)
                            except (ValueError, TypeError):
                                pass

                    if valid:
                        valid = np.asarray(valid)

                        print(
                            f"{indent}  min={valid.min():.6f}, "
                            f"max={valid.max():.6f}, "
                            f"mean={valid.mean():.6f}"
                        )

                        print(
                            f"{indent}  first 10 numeric values:"
                        )
                        print(
                            f"{indent}  {valid[:10]}"
                        )

                except Exception as e:
                    print(f"{indent}  Could not inspect: {e}")

    else:
        try:
            arr = np.asarray(obj)

            print(
                "Shape:", arr.shape,
                "Dtype:", arr.dtype,
                "Size:", arr.size
            )

            flat = arr.flatten()

            none_count = sum(x is None for x in flat)
            print("None count:", none_count)

            valid = []

            for x in flat:
                if x is not None:
                    try:
                        fx = float(x)
                        if np.isfinite(fx):
                            valid.append(fx)
                    except (ValueError, TypeError):
                        pass

            if valid:
                valid = np.asarray(valid)

                print("Min:", valid.min())
                print("Max:", valid.max())
                print("Mean:", valid.mean())
                print("First 10:", valid[:10])

        except Exception as e:
            print("Could not inspect:", e)


for filename in FILES:

    filepath = trial_dir / filename

    print("\n" + "=" * 80)

    if not filepath.exists():
        print("MISSING:", filename)
        continue

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    inspect_object(data, filename)


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)