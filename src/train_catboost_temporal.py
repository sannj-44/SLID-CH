from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data_processed" / "slid_ch_features_temporal.csv"
)
SPLIT_PATH = ROOT / "results" / "trial_split.json"

MODEL_DIR = ROOT / "models" / "catboost_temporal"
RESULT_DIR = ROOT / "results" / "catboost_temporal"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "catboost_temporal_model.cbm"

RANDOM_SEED = 42
THRESHOLD = 0.5

# ============================================================
# 2. LOAD THE EXISTING TRAIN/TEST SPLIT
# ============================================================

print("=" * 70)
print("CATBOOST TEMPORAL FEATURE EXPERIMENT")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Trial split not found: {SPLIT_PATH}"
    )

with open(SPLIT_PATH, "r", encoding="utf-8") as file:
    split_info = json.load(file)

train_trials = set(split_info["train_trials"])
test_trials = set(split_info["test_trials"])

overlap = train_trials.intersection(test_trials)

if overlap:
    raise ValueError(
        f"Train/test trial leakage detected: "
        f"{len(overlap)} shared trials."
    )

print(f"Training trials: {len(train_trials)}")
print(f"Testing trials:  {len(test_trials)}")
print("Shared trials:   0")

# ============================================================
# 3. IDENTIFY THE 288 INPUT FEATURES
# ============================================================

print("\nLoading temporal feature dataset column names...")

column_names = pd.read_csv(
    DATA_PATH,
    nrows=0
).columns.tolist()

# ------------------------------------------------------------
# 3A. Original IMU statistical features: 72
# ------------------------------------------------------------

original_suffixes = (
    "_mean",
    "_std",
    "_min",
    "_max",
    "_range",
    "_rms",
)

original_features = [
    column
    for column in column_names
    if column.endswith(original_suffixes)
]

# ------------------------------------------------------------
# 3B. First-difference features: 72
#
# IMPORTANT:
# "_abs_delta" also ends with "_delta".
# Therefore, explicitly exclude absolute-delta features.
# ------------------------------------------------------------

delta_features = [
    column
    for column in column_names
    if column.endswith("_delta")
    and not column.endswith("_abs_delta")
]

# ------------------------------------------------------------
# 3C. Absolute-difference features: 72
# ------------------------------------------------------------

absolute_delta_features = [
    column
    for column in column_names
    if column.endswith("_abs_delta")
]

# ------------------------------------------------------------
# 3D. Change-rate features: 72
# ------------------------------------------------------------

change_rate_features = [
    column
    for column in column_names
    if column.endswith("_change_rate")
]

# Combine the four feature groups.
feature_columns = (
    original_features
    + delta_features
    + absolute_delta_features
    + change_rate_features
)

# Remove duplicate column names while preserving order.
feature_columns = list(
    dict.fromkeys(feature_columns)
)

print(f"Original IMU features: {len(original_features)}")
print(f"Delta features:        {len(delta_features)}")
print(
    f"Absolute delta:        "
    f"{len(absolute_delta_features)}"
)
print(
    f"Change-rate features:  "
    f"{len(change_rate_features)}"
)
print(f"Total unique features: {len(feature_columns)}")

# ------------------------------------------------------------
# 3E. Validate the feature counts
# ------------------------------------------------------------

expected_counts = {
    "original IMU": (original_features, 72),
    "delta": (delta_features, 72),
    "absolute delta": (absolute_delta_features, 72),
    "change rate": (change_rate_features, 72),
}

for feature_type, (features, expected_count) in expected_counts.items():
    if len(features) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {feature_type} features, "
            f"but found {len(features)}."
        )

if len(feature_columns) != 288:
    raise ValueError(
        f"Expected 288 unique features, "
        f"but found {len(feature_columns)}."
    )

required_columns = {"trial", "slip"}

missing_required = required_columns.difference(
    column_names
)

if missing_required:
    raise ValueError(
        f"Required columns are missing: {missing_required}"
    )

# Explicitly prohibit target, metadata, and label-derived
# variables from being used as model inputs.
forbidden_features = {
    "slip",
    "max_contact_speed",
    "slip_fraction",
    "time_start",
    "time_end",
    "window_duration",
    "object",
    "trial",
}

leaked_columns = forbidden_features.intersection(
    feature_columns
)

if leaked_columns:
    raise ValueError(
        f"Potential leakage or metadata in input features: "
        f"{leaked_columns}"
    )

print("\nFeature validation passed.")
print("Target and label-derived variables are excluded.")

# ============================================================
# 4. LOAD ONLY THE REQUIRED DATA
# ============================================================

print("\nLoading dataset...")

use_columns = feature_columns + ["trial", "slip"]

# Use float32 for feature storage to reduce memory usage.
dtype_map = {
    column: np.float32
    for column in feature_columns
}

dtype_map["trial"] = str
dtype_map["slip"] = np.int8

df = pd.read_csv(
    DATA_PATH,
    usecols=use_columns,
    dtype=dtype_map,
)

print(f"Dataset rows loaded: {len(df):,}")

# ============================================================
# 5. VALIDATE DATA AND APPLY THE SAVED TRIAL SPLIT
# ============================================================

if df["trial"].isna().any():
    raise ValueError(
        "Missing trial identifiers detected."
    )

if df["slip"].isna().any():
    raise ValueError(
        "Missing target labels detected."
    )

observed_labels = set(
    df["slip"].unique()
)

if not observed_labels.issubset({0, 1}):
    raise ValueError(
        f"Expected binary labels 0 and 1; "
        f"found {observed_labels}."
    )

dataset_trials = set(
    df["trial"].unique()
)

unassigned_trials = (
    dataset_trials - train_trials - test_trials
)

if unassigned_trials:
    raise ValueError(
        f"{len(unassigned_trials)} trials are absent "
        f"from the saved split. "
        f"Examples: {list(unassigned_trials)[:5]}"
    )

train_mask = df["trial"].isin(
    train_trials
).to_numpy()

test_mask = df["trial"].isin(
    test_trials
).to_numpy()

if np.any(train_mask & test_mask):
    raise ValueError(
        "A row was assigned to both training and testing."
    )

if not np.all(train_mask | test_mask):
    raise ValueError(
        "Some rows were not assigned to training or testing."
    )

# Extract labels.
y_all = df["slip"].to_numpy(
    dtype=np.int8,
    copy=True,
)

# Extract all model features as float32.
X_all = df[feature_columns].to_numpy(
    dtype=np.float32,
    copy=True,
)

# Validate all model inputs.
if not np.isfinite(X_all).all():
    raise ValueError(
        "Input features contain NaN or infinite values. "
        "Resolve these before training."
    )

# Create train/test arrays using the existing split.
X_train = X_all[train_mask]
X_test = X_all[test_mask]

y_train = y_all[train_mask]
y_test = y_all[test_mask]

# Release large temporary arrays.
del X_all, y_all, df
gc.collect()

print("\nSplit validation passed.")
print(f"Training windows: {len(y_train):,}")
print(f"Testing windows:  {len(y_test):,}")
print(f"Training features: {X_train.shape[1]}")
print(f"Testing features:  {X_test.shape[1]}")

print("\nTraining class distribution:")
print(f"No Slip (0): {np.sum(y_train == 0):,}")
print(f"Slip (1):    {np.sum(y_train == 1):,}")

print("\nTesting class distribution:")
print(f"No Slip (0): {np.sum(y_test == 0):,}")
print(f"Slip (1):    {np.sum(y_test == 1):,}")

# ============================================================
# 6. INITIALIZE THE TUNED CATBOOST MODEL
# ============================================================

print("\nInitializing tuned CatBoost model...")

model = CatBoostClassifier(
    iterations=600,
    depth=8,
    learning_rate=0.1,
    l2_leaf_reg=5,
    random_strength=0.0,
    border_count=64,
    loss_function="Logloss",
    eval_metric="Logloss",
    auto_class_weights="Balanced",
    random_seed=RANDOM_SEED,
    verbose=100,
    thread_count=-1,
    allow_writing_files=False,
)

# ============================================================
# 7. TRAIN THE MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CATBOOST WITH TEMPORAL FEATURES")
print("=" * 70)

start_time = time.time()

model.fit(
    X_train,
    y_train,
)

training_time = time.time() - start_time

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds."
)

# Save the trained model.
model.save_model(
    str(MODEL_PATH)
)

print(f"Model saved to: {MODEL_PATH}")

# ============================================================
# 8. PREDICT ON HELD-OUT TEST TRIALS
# ============================================================

print("\nEvaluating on the held-out test trials...")

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(np.int8)

# ============================================================
# 9. CALCULATE EVALUATION METRICS
# ============================================================

metrics = {
    "model": "CatBoost with Temporal Features",
    "feature_count": int(X_train.shape[1]),
    "training_windows": int(len(y_train)),
    "testing_windows": int(len(y_test)),
    "training_trials": int(len(train_trials)),
    "testing_trials": int(len(test_trials)),
    "threshold": THRESHOLD,
    "training_time_seconds": round(
        training_time, 4
    ),
    "accuracy": float(
        accuracy_score(y_test, predictions)
    ),
    "precision": float(
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),
    "f1_score": float(
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_test,
            probabilities,
        )
    ),
    "pr_auc": float(
        average_precision_score(
            y_test,
            probabilities,
        )
    ),
}

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

for metric_name, metric_value in metrics.items():
    if isinstance(metric_value, float):
        print(
            f"{metric_name}: {metric_value:.6f}"
        )
    else:
        print(
            f"{metric_name}: {metric_value}"
        )

# Save metrics.
with open(
    RESULT_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=4,
    )

# ============================================================
# 10. CONFUSION MATRIX AND CLASSIFICATION REPORT
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

print("\nConfusion Matrix:")
print(cm)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    zero_division=0,
)

print("\nClassification Report:")
print(report)

with open(
    RESULT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

# Save the confusion matrix as CSV.
confusion_df = pd.DataFrame(
    cm,
    index=[
        "Actual No Slip",
        "Actual Slip",
    ],
    columns=[
        "Predicted No Slip",
        "Predicted Slip",
    ],
)

confusion_df.to_csv(
    RESULT_DIR / "confusion_matrix.csv"
)

# Plot the confusion matrix.
fig, ax = plt.subplots(
    figsize=(7, 6)
)

image = ax.imshow(
    cm,
    interpolation="nearest",
)

fig.colorbar(
    image,
    ax=ax,
)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["No Slip", "Slip"],
    yticklabels=["No Slip", "Slip"],
    xlabel="Predicted Label",
    ylabel="Actual Label",
    title="CatBoost Temporal Features - Confusion Matrix",
)

for row in range(2):
    for col in range(2):
        ax.text(
            col,
            row,
            f"{cm[row, col]:,}",
            ha="center",
            va="center",
        )

fig.tight_layout()

fig.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

importance_values = model.get_feature_importance()

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance_values,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

importance_df.to_csv(
    RESULT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTop 20 important features:")
print(
    importance_df.head(20).to_string(
        index=False
    )
)

# Plot the 20 most important features.
top_features = importance_df.head(
    20
).sort_values(
    "importance",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.barh(
    top_features["feature"],
    top_features["importance"],
)

ax.set_xlabel(
    "Feature Importance"
)

ax.set_title(
    "Top 20 CatBoost Temporal Features"
)

fig.tight_layout()

fig.savefig(
    RESULT_DIR / "feature_importance.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "actual_slip": y_test,
    "predicted_slip": predictions,
    "slip_probability": probabilities,
})

prediction_df.to_csv(
    RESULT_DIR / "test_predictions.csv",
    index=False,
)

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

print(f"Model: {MODEL_PATH}")
print(f"Results: {RESULT_DIR}")

print("\nKey metrics:")
print(f"Accuracy:  {metrics['accuracy']:.4%}")
print(f"Precision: {metrics['precision']:.4%}")
print(f"Recall:    {metrics['recall']:.4%}")
print(f"F1-score:  {metrics['f1_score']:.4%}")
print(f"ROC-AUC:   {metrics['roc_auc']:.4%}")
print(f"PR-AUC:    {metrics['pr_auc']:.4%}")

print("\nNext step: compare with the tuned CatBoost baseline.")