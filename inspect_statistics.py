import sys
import pickle
from pathlib import Path

sys.path.insert(0, r"IMUSlip\IMUSlip\code")

from slipnet import SlipDataset

import __main__
__main__.SlipDataset = SlipDataset

import torch

data_path = Path(r"IMUSlip\IMUSlip\data\1-Datasets\0-primary")

for name in ["train_data.pkl", "val_data.pkl", "test_data.pkl"]:

    with open(data_path / name, "rb") as f:
        data = pickle.load(f)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Class distribution
    labels = data.y_data.squeeze()

    unique, counts = torch.unique(labels, return_counts=True)

    print("\nClass distribution:")
    for label, count in zip(unique.tolist(), counts.tolist()):
        percentage = 100 * count / len(labels)
        print(f"Class {label}: {count:,} ({percentage:.2f}%)")

    # Reshape into samples × time × channels
    X = data.x_data.view(-1, 30, 12)

    print("\nReshaped X:", X.shape)

    print("\nOverall statistics:")
    print(f"Minimum : {X.min().item():.4f}")
    print(f"Maximum : {X.max().item():.4f}")
    print(f"Mean    : {X.mean().item():.4f}")
    print(f"Std     : {X.std().item():.4f}")

    print("\nChannel statistics:")

    for channel in range(12):
        values = X[:, :, channel]

        print(
            f"Channel {channel + 1:2d}: "
            f"min={values.min().item():8.4f}, "
            f"max={values.max().item():8.4f}, "
            f"mean={values.mean().item():8.4f}, "
            f"std={values.std().item():8.4f}"
        )