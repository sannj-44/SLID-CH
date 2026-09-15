from pathlib import Path


DATASET = Path(
    r"IMUSlip\IMUSlip\data\0-Trials"
)

files = sorted(DATASET.rglob("*"))

print("=" * 70)
print("RAW DATASET FILE STRUCTURE")
print("=" * 70)

# First trial directory
imu_file = next(DATASET.rglob("IMU_data.pkl"))
trial_dir = imu_file.parent

print("\nTrial:")
print(trial_dir)

print("\nFiles in this trial:")
for file in sorted(trial_dir.iterdir()):
    print(" ", file.name)