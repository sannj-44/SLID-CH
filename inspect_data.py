import sys
import pickle
from pathlib import Path

sys.path.insert(0, r"IMUSlip\IMUSlip\code")

from slipnet import SlipDataset

import __main__
__main__.SlipDataset = SlipDataset

data_path = Path(r"IMUSlip\IMUSlip\data\1-Datasets\0-primary")

with open(data_path / "train_data.pkl", "rb") as f:
    data = pickle.load(f)

print("=" * 60)
print("DATASET")
print("=" * 60)

print("Dataset type:", type(data))
print("Number of samples:", len(data))

print("\nX:")
print("Shape:", data.x_data.shape)
print("Data type:", data.x_data.dtype)

print("\nY:")
print("Shape:", data.y_data.shape)
print("Data type:", data.y_data.dtype)

print("\nFirst sample:")
print(data.x_data[0])

print("\nFirst label:")
print(data.y_data[0])

print("\nUnique labels:")
print(data.y_data.unique())

print("\nMean shape:")
print(data.mean.shape)

print("\nStd shape:")
print(data.std.shape)