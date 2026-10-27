"""
SLID-CH: Decision Tree for Robotic Fingertip Slip Detection

Features:
    72 IMU statistical features from left and right IMUs.

Target:
    slip (0 = No Slip, 1 = Slip)

Features:
    - Reuses the trial split from Logistic Regression.
    - Prevents leakage between training and test trials.
    - Excludes label-derived columns from model inputs.
    - Saves the trained model and evaluation results.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
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


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "slid_ch_features.csv"
SPLIT_PATH = PROJECT_ROOT / "results" / "trial_split.json"

OUTPUT_DIR = PROJECT_ROOT / "results" / "decision_tree"
MODEL_DIR = PROJECT_ROOT / "models" / "decision_tree"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Prevent the tree from growing excessively large.
# We can tune these parameters later.
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 20


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 65)
print("SLID-CH — DECISION TREE")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n"
        "Run prepare_dataset.py first."
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Total windows: {len(df):,}")


# ============================================================
# 3. VALIDATE DATA
# ============================================================

required_columns = {"slip", "trial"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if df["slip"].isna().any() or df["trial"].isna().any():
    raise ValueError("Missing values found in target or trial columns.")

df["slip"] = pd.to_numeric(
    df["slip"], errors="raise"
).astype(int)

if not set(df["slip"].unique()).issubset({0, 1}):
    raise ValueError("The slip target must contain only 0 and 1.")


# ============================================================
# 4. SELECT IMU FEATURES
# ============================================================

# Select only the 72 statistical IMU features.
# Do not include slip_fraction or max_contact_speed because
# these are derived from the labels.

feature_suffixes = (
    "_mean",
    "_std",
    "_min",
    "_max",
    "_range",
    "_rms",
)

feature_columns = [
    col for col in df.columns
    if col.endswith(feature_suffixes)
]

if not feature_columns:
    raise ValueError("No IMU statistical features found.")

if len(feature_columns) != 72:
    raise ValueError(
        f"Expected 72 IMU features, found {len(feature_columns)}."
    )

X = df[feature_columns].apply(
    pd.to_numeric, errors="raise"
)

y = df["slip"]

groups = df["trial"]

if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
    raise ValueError("IMU features contain NaN or infinite values.")

print(f"\nNumber of IMU features: {len(feature_columns)}")


# ============================================================
# 5. REUSE THE EXISTING TRIAL SPLIT
# ============================================================

# Logistic Regression created this file.
# Reusing it keeps both models' test sets identical.

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Trial split not found:\n{SPLIT_PATH}\n"
        "Run train_logistic_regression.py first to create it."
    )

with open(SPLIT_PATH, "r", encoding="utf-8") as file:
    split_info = json.load(file)

train_trials = set(
    str(value) for value in split_info["train_trials"]
)

test_trials = set(
    str(value) for value in split_info["test_trials"]
)

trial_ids = groups.astype(str)

if train_trials.intersection(test_trials):
    raise RuntimeError("Training and test trials overlap.")

known_trials = train_trials.union(test_trials)
actual_trials = set(trial_ids.unique())

if known_trials != actual_trials:
    missing_trials = actual_trials - known_trials
    unknown_trials = known_trials - actual_trials

    raise ValueError(
        "The current dataset does not match the saved trial split.\n"
        f"Trials missing from split: {sorted(missing_trials)[:10]}\n"
        f"Trials absent from dataset: {sorted(unknown_trials)[:10]}\n"
        "Check whether the dataset was regenerated or changed."
    )

train_mask = trial_ids.isin(train_trials)
test_mask = trial_ids.isin(test_trials)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\n" + "=" * 65)
print("TRAIN/TEST SPLIT")
print("=" * 65)

print(f"Training trials: {len(train_trials)}")
print(f"Testing trials:  {len(test_trials)}")
print(f"Training windows: {len(X_train):,}")
print(f"Testing windows:  {len(X_test):,}")

if set(trial_ids.loc[train_mask]).intersection(
    set(trial_ids.loc[test_mask])
):
    raise RuntimeError("Trial leakage detected.")

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError("Training and test data must contain both classes.")


# ============================================================
# 6. BUILD DECISION TREE
# ============================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n" + "=" * 65)
print("TRAINING DECISION TREE")
print("=" * 65)

start_time = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time

print(f"\nTraining time: {training_time:.2f} seconds")


# ============================================================
# 8. PREDICT TEST SET
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

metrics = {
    "model": "Decision Tree",
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
    "max_depth": MAX_DEPTH,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
}

print("\n" + "=" * 65)
print("DECISION TREE TEST RESULTS")
print("=" * 65)

for name, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{name:28s}: {value:.4f}")
    else:
        print(f"{name:28s}: {value}")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    zero_division=0,
)

print("\nClassification Report:")
print(report)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 10. SAVE METRICS AND REPORT
# ============================================================

pd.DataFrame([metrics]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            key: (
                float(value)
                if isinstance(value, (np.floating, np.integer))
                else value
            )
            for key, value in metrics.items()
        },
        file,
        indent=4,
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
).to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)


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

ax.set_title("Decision Tree — Test Set Confusion Matrix")

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

predictions = pd.DataFrame({
    "trial": groups.loc[test_mask].to_numpy(),
    "actual_slip": y_test.to_numpy(),
    "predicted_slip": y_pred,
    "slip_probability": y_probability,
})

if "object" in df.columns:
    predictions.insert(
        1,
        "object",
        df.loc[test_mask, "object"].to_numpy(),
    )

predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model_path = MODEL_DIR / "decision_tree.joblib"

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": "slip",
        "label_mapping": {
            0: "No Slip",
            1: "Slip",
        },
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
    },
    model_path,
)


# ============================================================
# 14. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

feature_importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTop 15 important features:")
print(feature_importance.head(15).to_string(index=False))


# ============================================================
# 15. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DECISION TREE COMPLETED")
print("=" * 65)

print(f"\nModel saved to:\n{model_path}")
print(f"\nResults saved in:\n{OUTPUT_DIR}")

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {path.name}")

print(
    "\nNote: Metrics measure agreement with the current dataset labels. "
    "Verify the intended SLID-CH label definition before interpreting "
    "these values as definitive real-world slip-detection performance."
)