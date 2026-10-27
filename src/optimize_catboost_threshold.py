from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier

from sklearn.model_selection import GroupKFold
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
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT
    / "data_processed"
    / "slid_ch_features_temporal.csv"
)

SPLIT_PATH = (
    ROOT
    / "results"
    / "trial_split.json"
)

SELECTED_FEATURES_PATH = (
    ROOT
    / "results"
    / "catboost_feature_selected"
    / "selected_features.json"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "catboost_feature_selected"
    / "catboost_selected_150_features.cbm"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "catboost_threshold_optimization"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

OOF_PATH = RESULT_DIR / "oof_predictions.csv"

OOF_METRICS_PATH = RESULT_DIR / "oof_threshold_comparison.csv"

TEST_METRICS_PATH = RESULT_DIR / "test_threshold_comparison.csv"

TEST_PREDICTIONS_PATH = RESULT_DIR / "test_predictions.csv"

SUMMARY_PATH = RESULT_DIR / "threshold_optimization_summary.json"

OOF_CONFUSION_PATH = RESULT_DIR / "oof_confusion_matrices.json"

TEST_CONFUSION_PATH = RESULT_DIR / "test_confusion_matrices.json"

PLOT_PATH = RESULT_DIR / "threshold_vs_metrics.png"

RANDOM_SEED = 42
N_SPLITS = 5

DEFAULT_THRESHOLD = 0.50

# Search thresholds from 0.10 through 0.90 in increments of 0.01.
THRESHOLDS = np.round(
    np.arange(0.10, 0.901, 0.01),
    2,
)

# Parameters from the tuned CatBoost model.
CATBOOST_PARAMS = {
    "iterations": 600,
    "depth": 8,
    "learning_rate": 0.1,
    "l2_leaf_reg": 5,
    "random_strength": 0.0,
    "border_count": 64,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "thread_count": -1,
    "allow_writing_files": False,
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate classification metrics using a specified threshold.
    """

    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(np.int8)

    y_true = np.asarray(y_true).astype(np.int8)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }


def calculate_confusion_matrix(y_true, probabilities, threshold):
    """
    Return a confusion matrix in the order:
    [[TN, FP],
     [FN, TP]]
    """

    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(np.int8)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        "threshold": float(threshold),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
        "matrix": cm.tolist(),
    }


def print_metrics(title, metrics):
    """
    Print a readable metrics summary.
    """

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    for name in [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]:
        if name in metrics:
            print(f"{name:12s}: {metrics[name]:.6f}")


# ============================================================
# 3. VALIDATE INPUT FILES
# ============================================================

print("=" * 75)
print("CATBOOST THRESHOLD OPTIMIZATION")
print("=" * 75)

required_files = [
    DATA_PATH,
    SPLIT_PATH,
    SELECTED_FEATURES_PATH,
    MODEL_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nAll required files exist.")


# ============================================================
# 4. LOAD THE SAVED TRIAL SPLIT
# ============================================================

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8",
) as file:
    split_info = json.load(file)

train_trials = set(
    str(trial)
    for trial in split_info["train_trials"]
)

test_trials = set(
    str(trial)
    for trial in split_info["test_trials"]
)

if train_trials.intersection(test_trials):
    raise ValueError(
        "Trial leakage detected: training and testing trials overlap."
    )

print(f"\nTraining trials: {len(train_trials)}")
print(f"Testing trials:  {len(test_trials)}")

if len(train_trials) != 156:
    print(
        "WARNING: Expected 156 training trials, "
        f"but found {len(train_trials)}."
    )

if len(test_trials) != 39:
    print(
        "WARNING: Expected 39 testing trials, "
        f"but found {len(test_trials)}."
    )


# ============================================================
# 5. LOAD THE SELECTED FEATURE LIST
# ============================================================

with open(
    SELECTED_FEATURES_PATH,
    "r",
    encoding="utf-8",
) as file:
    feature_info = json.load(file)

selected_features = feature_info["selected_features"]

expected_feature_count = int(
    feature_info["selected_feature_count"]
)

print(
    f"\nSelected features recorded in JSON: "
    f"{len(selected_features)}"
)

print(
    f"Expected selected feature count: "
    f"{expected_feature_count}"
)


# ============================================================
# 6. LOAD THE SAVED CATBOOST MODEL
# ============================================================

print("\nLoading the saved CatBoost model...")

model = CatBoostClassifier()

model.load_model(str(MODEL_PATH))

model_feature_names = model.feature_names_

if len(selected_features) != len(model_feature_names):
    raise ValueError(
        "Feature-count mismatch: selected_features.json contains "
        f"{len(selected_features)} features, but the model expects "
        f"{len(model_feature_names)} features."
    )

if len(selected_features) != expected_feature_count:
    raise ValueError(
        f"Feature-count mismatch: JSON declares "
        f"{expected_feature_count} features, but contains "
        f"{len(selected_features)}."
    )

if len(set(selected_features)) != len(selected_features):
    raise ValueError(
        "Duplicate feature names found in selected_features.json."
    )

if expected_feature_count != 150:
    raise ValueError(
        "This script expects the saved 150-feature model, "
        f"but the JSON declares {expected_feature_count} features."
    )

# The original feature-selection script trained the final model
# using a NumPy array:
#
# X_train_selected = X_train[:, selected_feature_indices]
#
# Therefore, CatBoost may store numeric feature names such as
# "0", "1", "2", ... rather than the original column names.
#
# The original training script constructed selected_feature_indices
# from selected_features, preserving the order in that JSON file.
#
# We therefore reconstruct the feature matrix using that exact order.

print(
    "\nFeature mapping verified using the original training procedure: "
    f"{len(selected_features)} selected features."
)

print(
    "Feature order source: "
    "results/catboost_feature_selected/selected_features.json"
)

print(
    f"Model feature count: {len(model_feature_names)}"
)

print(
    f"Stored CatBoost feature-name examples: "
    f"{model_feature_names[:5]}"
)


# ============================================================
# 7. VALIDATE FEATURE COLUMNS
# ============================================================

print("\nChecking the dataset columns...")

available_columns = pd.read_csv(
    DATA_PATH,
    nrows=0,
).columns.tolist()

missing_features = [
    feature
    for feature in selected_features
    if feature not in available_columns
]

if missing_features:
    raise ValueError(
        "Selected features missing from the dataset:\n"
        + "\n".join(missing_features[:20])
    )

# These columns must never be model inputs.
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

forbidden_used = (
    set(selected_features)
    .intersection(forbidden_features)
)

if forbidden_used:
    raise ValueError(
        "Forbidden target-derived or metadata columns found "
        f"in the selected features: {sorted(forbidden_used)}"
    )

print("Selected feature columns validated.")


# ============================================================
# 8. LOAD THE DATASET
# ============================================================

print("\nLoading selected features and labels...")

use_columns = list(
    dict.fromkeys(
        selected_features + ["trial", "slip"]
    )
)

dtype_map = {
    feature: np.float32
    for feature in selected_features
}

dtype_map["trial"] = str
dtype_map["slip"] = np.int8

df = pd.read_csv(
    DATA_PATH,
    usecols=use_columns,
    dtype=dtype_map,
)

if df.empty:
    raise ValueError("The dataset is empty.")

if df[selected_features].isna().any().any():
    raise ValueError(
        "Missing values found in the selected features."
    )

X_check = df[selected_features].to_numpy(
    dtype=np.float32
)

if not np.isfinite(X_check).all():
    raise ValueError(
        "NaN or infinite values found in the selected features."
    )

del X_check
gc.collect()

if df["slip"].isna().any():
    raise ValueError("Missing target labels detected.")

if not set(df["slip"].unique()).issubset({0, 1}):
    raise ValueError(
        "The target column must contain only binary labels 0 and 1."
    )

dataset_trials = set(
    df["trial"].astype(str).unique()
)

unassigned_trials = (
    dataset_trials - train_trials - test_trials
)

if unassigned_trials:
    raise ValueError(
        "Some dataset trials are missing from trial_split.json: "
        f"{list(unassigned_trials)[:10]}"
    )

missing_train_trials = train_trials - dataset_trials
missing_test_trials = test_trials - dataset_trials

if missing_train_trials:
    raise ValueError(
        "Training trials missing from the dataset: "
        f"{list(missing_train_trials)[:10]}"
    )

if missing_test_trials:
    raise ValueError(
        "Testing trials missing from the dataset: "
        f"{list(missing_test_trials)[:10]}"
    )

print(f"Dataset windows: {len(df):,}")
print(f"Selected features: {len(selected_features)}")


# ============================================================
# 9. CREATE TRAINING AND TEST MATRICES
# ============================================================

print("\nCreating training and testing matrices...")

trial_values = df["trial"].astype(str)

train_mask = trial_values.isin(
    train_trials
).to_numpy()

test_mask = trial_values.isin(
    test_trials
).to_numpy()

if np.any(train_mask & test_mask):
    raise ValueError(
        "Training and test rows overlap."
    )

if not np.all(train_mask | test_mask):
    raise ValueError(
        "Some dataset rows were not assigned to training or testing."
    )

# IMPORTANT:
# Use selected_features in precisely the saved JSON order.
# Do not use the numeric feature names stored by CatBoost
# as pandas column names.

X_all = df[selected_features].to_numpy(
    dtype=np.float32,
    copy=True,
)

y_all = df["slip"].to_numpy(
    dtype=np.int8,
    copy=True,
)

groups_all = trial_values.to_numpy(
    copy=True
)

X_train = X_all[train_mask]
y_train = y_all[train_mask]
groups_train = groups_all[train_mask]

X_test = X_all[test_mask]
y_test = y_all[test_mask]

test_trial_values = groups_all[test_mask]

del X_all, y_all, groups_all, df
gc.collect()

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError(
        "Training or test matrix is empty."
    )

if X_train.shape[1] != len(model_feature_names):
    raise ValueError(
        f"Training matrix has {X_train.shape[1]} features, "
        f"but the model expects {len(model_feature_names)}."
    )

if X_test.shape[1] != len(model_feature_names):
    raise ValueError(
        f"Test matrix has {X_test.shape[1]} features, "
        f"but the model expects {len(model_feature_names)}."
    )

if set(groups_train).intersection(set(test_trial_values)):
    raise ValueError(
        "Training/test trial overlap detected after matrix construction."
    )

print(f"Training matrix: {X_train.shape}")
print(f"Testing matrix:  {X_test.shape}")

print(
    f"Training label distribution: "
    f"No Slip={int(np.sum(y_train == 0)):,}, "
    f"Slip={int(np.sum(y_train == 1)):,}"
)

print(
    f"Testing label distribution: "
    f"No Slip={int(np.sum(y_test == 0)):,}, "
    f"Slip={int(np.sum(y_test == 1)):,}"
)


# ============================================================
# 10. GENERATE OUT-OF-FOLD TRAINING PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("GENERATING OUT-OF-FOLD PREDICTIONS")
print("=" * 75)

print(
    f"Using {N_SPLITS}-fold GroupKFold with trial-level grouping."
)

print(
    "Each training window receives a probability from a model "
    "that did not train on that window's trial."
)

cv = GroupKFold(
    n_splits=N_SPLITS
)

oof_probabilities = np.full(
    shape=len(y_train),
    fill_value=np.nan,
    dtype=np.float64,
)

oof_fold_numbers = np.full(
    shape=len(y_train),
    fill_value=-1,
    dtype=np.int8,
)

start_oof = time.time()

for fold_number, (fit_idx, val_idx) in enumerate(
    cv.split(
        X_train,
        y_train,
        groups_train,
    ),
    start=1,
):
    print("\n" + "-" * 65)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("-" * 65)

    X_fit = X_train[fit_idx]
    y_fit = y_train[fit_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    groups_fit = groups_train[fit_idx]
    groups_val = groups_train[val_idx]

    # Verify that no trial appears in both sides of this fold.
    if set(groups_fit).intersection(set(groups_val)):
        raise ValueError(
            f"Trial leakage detected in fold {fold_number}."
        )

    fold_model = CatBoostClassifier(
        **CATBOOST_PARAMS
    )

    fold_start = time.time()

    fold_model.fit(
        X_fit,
        y_fit,
    )

    fold_probabilities = fold_model.predict_proba(
        X_val
    )[:, 1]

    # CRITICAL:
    # Write predictions into the original training-array positions.
    # Never concatenate fold outputs directly because the fold order
    # differs from the original order of X_train and y_train.

    oof_probabilities[val_idx] = fold_probabilities
    oof_fold_numbers[val_idx] = fold_number

    fold_predictions = (
        fold_probabilities >= DEFAULT_THRESHOLD
    ).astype(np.int8)

    fold_f1 = f1_score(
        y_val,
        fold_predictions,
        zero_division=0,
    )

    fold_accuracy = accuracy_score(
        y_val,
        fold_predictions,
    )

    elapsed = time.time() - fold_start

    print(f"Training windows:   {len(y_fit):,}")
    print(f"Validation windows: {len(y_val):,}")
    print(f"Validation trials:  {len(set(groups_val))}")
    print(f"Accuracy at 0.50:   {fold_accuracy:.6f}")
    print(f"F1-score at 0.50:   {fold_f1:.6f}")
    print(f"Fold elapsed time:  {elapsed:.2f} seconds")

    del fold_model
    del X_fit, y_fit, X_val, y_val
    del groups_fit, groups_val
    del fold_probabilities, fold_predictions

    gc.collect()

oof_seconds = time.time() - start_oof

if np.isnan(oof_probabilities).any():
    raise RuntimeError(
        "Some training windows did not receive OOF predictions."
    )

if (oof_fold_numbers < 1).any():
    raise RuntimeError(
        "Some training windows do not have an assigned CV fold."
    )

if not np.isfinite(oof_probabilities).all():
    raise RuntimeError(
        "Invalid values found in OOF probabilities."
    )

if (
    (oof_probabilities < 0).any()
    or (oof_probabilities > 1).any()
):
    raise RuntimeError(
        "OOF probabilities fall outside the interval [0, 1]."
    )

print(
    f"\nOOF generation completed in "
    f"{oof_seconds / 60:.2f} minutes."
)


# ============================================================
# 11. SAVE ALIGNED OOF PREDICTIONS
# ============================================================

print("\nSaving OOF predictions...")

oof_df = pd.DataFrame({
    "trial": groups_train,
    "actual_slip": y_train,
    "oof_probability": oof_probabilities,
    "fold": oof_fold_numbers,
})

# The rows are aligned to the original X_train/y_train order.
# Keep the original training-array row position explicitly.
oof_df.insert(
    0,
    "train_row_index",
    np.arange(len(y_train)),
)

oof_df.to_csv(
    OOF_PATH,
    index=False,
)

print(f"OOF predictions saved to:\n{OOF_PATH}")


# ============================================================
# 12. EVALUATE THE OOF PREDICTIONS AT DIFFERENT THRESHOLDS
# ============================================================

print("\n" + "=" * 75)
print("OOF THRESHOLD COMPARISON")
print("=" * 75)

oof_records = []

for threshold in THRESHOLDS:
    metrics = calculate_metrics(
        y_train,
        oof_probabilities,
        float(threshold),
    )

    oof_records.append(metrics)

oof_metrics_df = pd.DataFrame(
    oof_records
)

oof_metrics_df.to_csv(
    OOF_METRICS_PATH,
    index=False,
)

# Select the threshold with the highest OOF F1-score.
#
# If multiple thresholds have the same F1-score, prefer the one
# closest to 0.50. This tie-break rule is deterministic.

best_f1 = oof_metrics_df["f1_score"].max()

best_candidates = oof_metrics_df[
    np.isclose(
        oof_metrics_df["f1_score"],
        best_f1,
        rtol=0,
        atol=1e-12,
    )
].copy()

best_candidates["distance_from_default"] = (
    best_candidates["threshold"] - DEFAULT_THRESHOLD
).abs()

best_candidates = best_candidates.sort_values(
    [
        "distance_from_default",
        "threshold",
    ],
    ascending=[True, True],
)

best_threshold = float(
    best_candidates.iloc[0]["threshold"]
)

best_oof_metrics = calculate_metrics(
    y_train,
    oof_probabilities,
    best_threshold,
)

default_oof_metrics = calculate_metrics(
    y_train,
    oof_probabilities,
    DEFAULT_THRESHOLD,
)

print_metrics(
    "OOF METRICS AT DEFAULT THRESHOLD",
    default_oof_metrics,
)

print_metrics(
    "OOF METRICS AT OPTIMIZED THRESHOLD",
    best_oof_metrics,
)

print(
    f"\nSelected threshold from OOF F1: "
    f"{best_threshold:.2f}"
)

print(
    "This threshold was selected using training-trial "
    "OOF predictions, not the held-out test labels."
)


# ============================================================
# 13. SAVE OOF CONFUSION MATRICES
# ============================================================

oof_confusion_matrices = {
    "default_threshold": calculate_confusion_matrix(
        y_train,
        oof_probabilities,
        DEFAULT_THRESHOLD,
    ),
    "optimized_threshold": calculate_confusion_matrix(
        y_train,
        oof_probabilities,
        best_threshold,
    ),
}

with open(
    OOF_CONFUSION_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        oof_confusion_matrices,
        file,
        indent=4,
    )


# ============================================================
# 14. EVALUATE THE SAVED MODEL ON HELD-OUT TEST TRIALS
# ============================================================

print("\n" + "=" * 75)
print("HELD-OUT TEST EVALUATION")
print("=" * 75)

print(
    "The held-out test set will be evaluated at both thresholds."
)

print(
    "The test set is not used to select the threshold."
)

# The saved model was fitted using the exact feature order stored
# in selected_features.json. X_test was reconstructed in that order.

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

if not np.isfinite(test_probabilities).all():
    raise RuntimeError(
        "Invalid probabilities produced by the saved model."
    )

default_test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
    DEFAULT_THRESHOLD,
)

optimized_test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
    best_threshold,
)

print_metrics(
    "HELD-OUT TEST RESULTS AT THRESHOLD 0.50",
    default_test_metrics,
)

print_metrics(
    f"HELD-OUT TEST RESULTS AT THRESHOLD {best_threshold:.2f}",
    optimized_test_metrics,
)


# ============================================================
# 15. SAVE TEST COMPARISON
# ============================================================

test_comparison_df = pd.DataFrame([
    {
        "model": "CatBoost selected 150 features",
        **default_test_metrics,
    },
    {
        "model": "CatBoost selected 150 features",
        **optimized_test_metrics,
    },
])

test_comparison_df.to_csv(
    TEST_METRICS_PATH,
    index=False,
)

print(
    f"\nTest threshold comparison saved to:\n"
    f"{TEST_METRICS_PATH}"
)


# ============================================================
# 16. SAVE TEST PREDICTIONS
# ============================================================

default_test_predictions = (
    test_probabilities >= DEFAULT_THRESHOLD
).astype(np.int8)

optimized_test_predictions = (
    test_probabilities >= best_threshold
).astype(np.int8)

test_predictions_df = pd.DataFrame({
    "trial": test_trial_values,
    "actual_slip": y_test,
    "slip_probability": test_probabilities,
    "prediction_threshold_050": default_test_predictions,
    "prediction_optimized_threshold": optimized_test_predictions,
})

test_predictions_df.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False,
)

print(
    f"Test predictions saved to:\n"
    f"{TEST_PREDICTIONS_PATH}"
)


# ============================================================
# 17. SAVE TEST CONFUSION MATRICES
# ============================================================

test_confusion_matrices = {
    "default_threshold": calculate_confusion_matrix(
        y_test,
        test_probabilities,
        DEFAULT_THRESHOLD,
    ),
    "optimized_threshold": calculate_confusion_matrix(
        y_test,
        test_probabilities,
        best_threshold,
    ),
}

with open(
    TEST_CONFUSION_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        test_confusion_matrices,
        file,
        indent=4,
    )

print("\nConfusion matrix at threshold 0.50:")
print(
    np.array(
        test_confusion_matrices[
            "default_threshold"
        ]["matrix"]
    )
)

print(
    f"\nConfusion matrix at threshold {best_threshold:.2f}:"
)

print(
    np.array(
        test_confusion_matrices[
            "optimized_threshold"
        ]["matrix"]
    )
)


# ============================================================
# 18. SAVE CLASSIFICATION REPORTS
# ============================================================

default_report = classification_report(
    y_test,
    default_test_predictions,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    zero_division=0,
)

optimized_report = classification_report(
    y_test,
    optimized_test_predictions,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    zero_division=0,
)

with open(
    RESULT_DIR / "classification_report_threshold_050.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(default_report)

with open(
    RESULT_DIR / "classification_report_optimized_threshold.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(optimized_report)

print("\nClassification report at threshold 0.50:")
print(default_report)

print(
    f"\nClassification report at threshold {best_threshold:.2f}:"
)
print(optimized_report)


# ============================================================
# 19. PLOT THRESHOLD VERSUS METRICS
# ============================================================

print("\nGenerating threshold comparison plot...")

fig, ax = plt.subplots(
    figsize=(11, 7)
)

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]:
    ax.plot(
        oof_metrics_df["threshold"],
        oof_metrics_df[metric],
        label=metric.replace("_", " ").title(),
    )

ax.axvline(
    x=DEFAULT_THRESHOLD,
    linestyle="--",
    label="Default threshold (0.50)",
)

ax.axvline(
    x=best_threshold,
    linestyle=":",
    label=f"Selected threshold ({best_threshold:.2f})",
)

ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Metric Score")
ax.set_title(
    "CatBoost OOF Threshold Optimization"
)

ax.set_xlim(
    float(THRESHOLDS.min()),
    float(THRESHOLDS.max()),
)

ax.set_ylim(0, 1)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PLOT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Threshold plot saved to:\n{PLOT_PATH}")


# ============================================================
# 20. SAVE FINAL SUMMARY
# ============================================================

summary = {
    "experiment": "CatBoost threshold optimization",
    "model_path": str(MODEL_PATH),
    "dataset_path": str(DATA_PATH),
    "trial_split_path": str(SPLIT_PATH),
    "selected_features_path": str(SELECTED_FEATURES_PATH),
    "selected_feature_count": len(selected_features),
    "feature_order_source": str(SELECTED_FEATURES_PATH),
    "training_trials": int(len(train_trials)),
    "testing_trials": int(len(test_trials)),
    "training_windows": int(len(y_train)),
    "testing_windows": int(len(y_test)),
    "cross_validation": {
        "method": "GroupKFold",
        "number_of_folds": N_SPLITS,
        "grouping": "trial",
        "oof_generation_seconds": round(
            oof_seconds,
            3,
        ),
    },
    "threshold_selection": {
        "method": "Maximum OOF F1-score",
        "threshold_search_min": float(THRESHOLDS.min()),
        "threshold_search_max": float(THRESHOLDS.max()),
        "threshold_search_step": 0.01,
        "tie_break": "Closest threshold to 0.50",
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_threshold": best_threshold,
        "oof_default_threshold_metrics": default_oof_metrics,
        "oof_selected_threshold_metrics": best_oof_metrics,
    },
    "held_out_test_results": {
        "default_threshold": default_test_metrics,
        "optimized_threshold": optimized_test_metrics,
    },
    "outputs": {
        "oof_predictions": str(OOF_PATH),
        "oof_threshold_comparison": str(OOF_METRICS_PATH),
        "test_threshold_comparison": str(TEST_METRICS_PATH),
        "test_predictions": str(TEST_PREDICTIONS_PATH),
        "oof_confusion_matrices": str(OOF_CONFUSION_PATH),
        "test_confusion_matrices": str(TEST_CONFUSION_PATH),
        "threshold_plot": str(PLOT_PATH),
    },
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 75)

print(f"Selected features:       {len(selected_features)}")
print(f"Training windows:        {len(y_train):,}")
print(f"Testing windows:         {len(y_test):,}")

print(
    f"Default threshold:       {DEFAULT_THRESHOLD:.2f}"
)

print(
    f"Selected threshold:      {best_threshold:.2f}"
)

print("\nHELD-OUT TEST COMPARISON")

print("-" * 75)

print(
    f"{'Metric':<15}"
    f"{'Threshold 0.50':>20}"
    f"{'Optimized':>20}"
)

print("-" * 75)

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]:
    default_value = default_test_metrics[metric]
    optimized_value = optimized_test_metrics[metric]

    print(
        f"{metric:<15}"
        f"{default_value:>20.6f}"
        f"{optimized_value:>20.6f}"
    )

print("-" * 75)

print(
    f"\nSummary saved to:\n{SUMMARY_PATH}"
)

print(
    f"\nAll outputs are in:\n{RESULT_DIR}"
)

print("\nImportant:")
print(
    "The optimized threshold was chosen using training-trial "
    "OOF predictions. Use the held-out test results to report "
    "its performance, not to further tune the threshold."
)

print("\nDone.")