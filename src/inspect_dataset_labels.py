from pathlib import Path
import pickle


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "IMUSlip"
    / "IMUSlip"
    / "data"
    / "0-Trials"
)


# ============================================================
# INSPECT PICKLE FILE
# ============================================================

def inspect_pickle(file_path):
    print("\n" + "-" * 70)
    print(f"FILE: {file_path.name}")
    print("-" * 70)

    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)

    except Exception as error:
        print(f"Could not read file: {error}")
        return

    print("Python type:", type(data).__name__)

    if isinstance(data, dict):

        print("Dictionary keys:", list(data.keys()))

        for key, value in data.items():

            print(f"\nKey: {key!r}")
            print("Value type:", type(value).__name__)

            if isinstance(value, dict):

                print("Nested keys:", list(value.keys()))

                for nested_key, nested_value in value.items():

                    try:
                        length = len(nested_value)
                    except TypeError:
                        length = "N/A"

                    print(
                        f"  {nested_key!r}: "
                        f"type={type(nested_value).__name__}, "
                        f"length={length}"
                    )

                    # Display a small sample of scalar data.
                    if isinstance(
                        nested_value,
                        (list, tuple),
                    ):
                        print(
                            "    First five values:",
                            nested_value[:5],
                        )

            elif isinstance(value, (list, tuple)):

                print("Length:", len(value))
                print("First five values:", value[:5])

            else:

                try:
                    print("Length:", len(value))
                except TypeError:
                    print("Value:", repr(value)[:200])

    elif isinstance(data, (list, tuple)):

        print("Length:", len(data))
        print("First five values:", data[:5])

    else:

        print("Value:", repr(data)[:500])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SLID-CH DATASET LABEL INSPECTION")
    print("=" * 70)

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"\nDataset directory not found:\n{DATA_ROOT}\n"
            "Check the location of your IMUSlip dataset."
        )

    # Find all trials containing IMU_data.pkl.
    trial_dirs = sorted(
        path
        for path in DATA_ROOT.rglob("*")
        if path.is_dir()
        and (path / "IMU_data.pkl").is_file()
    )

    print("Dataset directory:", DATA_ROOT)
    print("Trials found:", len(trial_dirs))

    if not trial_dirs:

        raise RuntimeError(
            "No trial directories containing IMU_data.pkl found."
        )

    # Inspect the first trial only.
    trial_dir = trial_dirs[0]

    print("\nExample trial:")
    print(trial_dir)

    print("\nFiles in this trial:")

    for path in sorted(trial_dir.iterdir()):

        if path.is_file():

            print(
                f"  {path.name} "
                f"({path.stat().st_size:,} bytes)"
            )

    # Inspect every pickle file in the example trial.
    pickle_files = sorted(
        trial_dir.glob("*.pkl")
    )

    if not pickle_files:

        print("No pickle files found in the example trial.")

    for file_path in pickle_files:

        inspect_pickle(file_path)

    # Count pickle filenames across all trials.
    print("\n" + "=" * 70)
    print("PICKLE FILES ACROSS ALL TRIALS")
    print("=" * 70)

    file_counts = {}

    for trial_dir in trial_dirs:

        for file_path in trial_dir.glob("*.pkl"):

            file_counts[file_path.name] = (
                file_counts.get(file_path.name, 0) + 1
            )

    for filename, count in sorted(file_counts.items()):

        print(f"{filename}: {count} trial(s)")

    print("\nInspection completed.")
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()