"""
SLID-CH: Logistic Regression for Robotic Fingertip Slip Detection

Features:
    72 IMU statistical features from left and right IMUs.

Target:
    slip (0 = no slip, 1 = slip)

Important:
    - Splits data by trial to prevent window-level leakage.
    - Excludes label-derived features and metadata from model inputs.
    - Saves the trained model, metrics, classification report,
      confusion matrix, and test predictions.
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("default")


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "slid_ch_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "logistic_regression"
MODEL_DIR = PROJECT_ROOT / "models" / "logistic_regression"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Maximum iterations allowed for Logistic Regression.
MAX_ITER = 1000


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 65)
print("SLID-CH — LOGISTIC REGRESSION")
print("=" * 65)

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Dataset: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n"
        "Check that data_processed/slid_ch_features.csv exists."
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Number of rows: {len(df):,}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {"slip", "trial"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {sorted(missing_columns)}"
    )

if df["slip"].isna().any():
    raise ValueError("The target column 'slip' contains missing values.")

if df["trial"].isna().any():
    raise ValueError("The 'trial' column contains missing values.")

# Convert target labels to integers.
df["slip"] = pd.to_numeric(df["slip"], errors="raise").astype(int)

if not set(df["slip"].unique()).issubset({0, 1}):
    raise ValueError(
        f"Expected binary target labels 0 and 1, "
        f"found: {sorted(df['slip'].unique())}"
    )

print("\nTarget distribution:")
print(df["slip"].value_counts().sort_index())

print("\nTarget distribution (%):")
print((df["slip"].value_counts(normalize=True).sort_index() * 100).round(2))


# ============================================================
# 4. SELECT IMU FEATURES — PREVENT DATA LEAKAGE
# ============================================================

# The preprocessing pipeline generates six statistical features
# for each of the twelve IMU channels.
#
# Allowed suffixes:
#   _mean, _std, _min, _max, _range, _rms
#
# Excluded from model inputs:
#   slip                 -> target
#   max_contact_speed    -> label-derived information
#   slip_fraction        -> label-derived information
#   time_start/end       -> timestamps
#   window_duration      -> metadata
#   object               -> metadata
#   trial                -> grouping identifier

feature_suffixes = (
    "_mean",
    "_std",
    "_min",
    "_max",
    "_range",
    "_rms",
)

feature_columns = [
    column
    for column in df.columns
    if column.endswith(feature_suffixes)
]

if not feature_columns:
    raise ValueError(
        "No IMU statistical features were found. "
        "Expected columns ending in _mean, _std, _min, _max, "
        "_range, or _rms."
    )

print(f"\nNumber of selected IMU features: {len(feature_columns)}")

print("\nSelected features:")
for feature in feature_columns:
    print(f"  {feature}")

if len(feature_columns) != 72:
    print(
        "\nWARNING: Expected 72 IMU features based on the current "
        f"preprocessing pipeline, but found {len(feature_columns)}."
    )

X = df[feature_columns].copy()
y = df["slip"].copy()
groups = df["trial"].copy()

# Ensure features are numeric.
X = X.apply(pd.to_numeric, errors="raise")

if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
    raise ValueError(
        "IMU features contain NaN or infinite values. "
        "Resolve these during preprocessing before training."
    )


# ============================================================
# 5. SPLIT BY TRIAL — NOT BY WINDOW
# ============================================================

# Windows from the same trial are related and may overlap.
# GroupShuffleSplit ensures that each trial belongs entirely
# to either the training set or the test set.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_trials = set(groups.iloc[train_indices])
test_trials = set(groups.iloc[test_indices])

overlap = train_trials.intersection(test_trials)

if overlap:
    raise RuntimeError(
        f"Trial leakage detected! Shared trials: {sorted(overlap)}"
    )

print("\n" + "=" * 65)
print("TRAIN/TEST SPLIT")
print("=" * 65)

print(f"Training windows: {len(X_train):,}")
print(f"Testing windows:  {len(X_test):,}")
print(f"Training trials:  {len(train_trials)}")
print(f"Testing trials:   {len(test_trials)}")
print(f"Shared trials:    {len(overlap)}")

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Both training and test sets must contain both classes."
    )

# Save the split definition so other model scripts can reproduce it.
split_info = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "train_trials": sorted(str(value) for value in train_trials),
    "test_trials": sorted(str(value) for value in test_trials),
    "train_window_count": int(len(X_train)),
    "test_window_count": int(len(X_test)),
}

split_path = PROJECT_ROOT / "results" / "trial_split.json"
split_path.parent.mkdir(parents=True, exist_ok=True)

with open(split_path, "w", encoding="utf-8") as file:
    json.dump(split_info, file, indent=4)

print(f"\nTrial split saved to: {split_path}")


# ============================================================
# 6. BUILD LOGISTIC REGRESSION PIPELINE
# ============================================================

# StandardScaler is fitted only on the training data.
# The pipeline applies the same transformation to test data.

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=MAX_ITER,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n" + "=" * 65)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 65)

start_time = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds.")


# ============================================================
# 8. PREDICT TEST SET
# ============================================================

print("\nGenerating test predictions...")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

metrics = {
    "model": "Logistic Regression",
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, y_probability),
    "pr_auc": average_precision_score(y_test, y_probability),
    "training_time_seconds": training_time,
    "train_windows": len(X_train),
    "test_windows": len(X_test),
    "train_trials": len(train_trials),
    "test_trials": len(test_trials),
    "number_of_features": len(feature_columns),
}

print("\n" + "=" * 65)
print("TEST SET RESULTS")
print("=" * 65)

for name, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{name:28s}: {value:.4f}")
    else:
        print(f"{name:28s}: {value}")

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    zero_division=0,
)

print(report)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 10. SAVE METRICS AND REPORT
# ============================================================

metrics_path = OUTPUT_DIR / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        {key: float(value) if isinstance(
            value, (np.floating, np.integer)
        ) else value for key, value in metrics.items()},
        file,
        indent=4,
    )

pd.DataFrame([metrics]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

pd.DataFrame(
    cm,
    index=["Actual No Slip", "Actual Slip"],
    columns=["Predicted No Slip", "Predicted Slip"],
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")


# ============================================================
# 11. SAVE CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Slip", "Slip"],
)

display.plot(
    ax=ax,
    values_format="d",
    colorbar=False,
)

ax.set_title("Logistic Regression — Test Set Confusion Matrix")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame(
    {
        "trial": groups.iloc[test_indices].to_numpy(),
        "actual_slip": y_test.to_numpy(),
        "predicted_slip": y_pred,
        "slip_probability": y_probability,
    }
)

# Include object metadata for later error analysis, if available.
if "object" in df.columns:
    predictions.insert(
        1,
        "object",
        df["object"].iloc[test_indices].to_numpy(),
    )

predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)


# ============================================================
# 13. SAVE TRAINED MODEL AND FEATURE NAMES
# ============================================================

model_path = MODEL_DIR / "logistic_regression.joblib"

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": "slip",
        "label_mapping": {
            0: "No Slip",
            1: "Slip",
        },
        "split_random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    },
    model_path,
)


# ============================================================
# 14. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION COMPLETED")
print("=" * 65)

print(f"\nModel saved to:\n{model_path}")

print(f"\nResults saved in:\n{OUTPUT_DIR}")

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {path.name}")

print("\nImportant:")
print(
    "These metrics measure agreement with the current dataset labels. "
    "The labels are based on the existing velocity-threshold procedure "
    "and should be verified against the intended SLID-CH label definition "
    "before interpreting the results as definitive slip-detection performance."
)