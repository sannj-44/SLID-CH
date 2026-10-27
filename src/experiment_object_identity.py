"""
SLID-CH Experiment 1: Add Object Identity to CatBoost

Compares:
    A. Existing selected-feature CatBoost model
    B. New CatBoost model using selected IMU features + object identity

Uses:
    - Existing selected 150 IMU features
    - Existing saved trial-level train/test split
    - Same tuned CatBoost hyperparameters
    - Object identity as a categorical feature

Outputs:
    results/object_identity_experiment/

The original dataset and existing model are not modified.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

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
)


warnings.filterwarnings("ignore", category=FutureWarning)


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

FEATURES_PATH = (
    ROOT
    / "results"
    / "catboost_feature_selected"
    / "selected_features.json"
)

BASELINE_MODEL_PATH = (
    ROOT
    / "models"
    / "catboost_feature_selected"
    / "catboost_selected_150_features.cbm"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "object_identity_experiment"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "catboost_object_identity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

# The existing tuned CatBoost configuration.
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

THRESHOLDS = [0.41, 0.50]

OBJECT_COLUMN = "object"
TARGET_COLUMN = "slip"
GROUP_COLUMN = "trial"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_selected_features(path):
    """Load the selected feature names from JSON."""

    with open(path, "r", encoding="utf-8") as file:
        content = json.load(file)

    if isinstance(content, list):
        features = content

    elif isinstance(content, dict):
        features = None

        for key in (
            "selected_features",
            "features",
            "top_features",
        ):
            if isinstance(content.get(key), list):
                features = content[key]
                break

        if features is None:
            raise ValueError(
                "Cannot identify the feature list in selected_features.json."
            )

    else:
        raise ValueError(
            "Unsupported selected_features.json format."
        )

    if len(features) != 150:
        raise ValueError(
            f"Expected 150 selected features; found {len(features)}."
        )

    if len(features) != len(set(features)):
        raise ValueError("Duplicate selected features detected.")

    return features


def evaluate_predictions(y_true, probabilities, threshold):
    """Calculate binary classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

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
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_json(data, path):
    """Save a dictionary as JSON."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def make_model():
    """Create a fresh CatBoost model."""

    return CatBoostClassifier(**CATBOOST_PARAMS)


# ============================================================
# 3. LOAD AND VALIDATE INPUT FILES
# ============================================================

print("=" * 78)
print("EXPERIMENT 1: CATBOOST WITH OBJECT IDENTITY")
print("=" * 78)

for path in (
    DATA_PATH,
    SPLIT_PATH,
    FEATURES_PATH,
    BASELINE_MODEL_PATH,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

required_columns = {
    OBJECT_COLUMN,
    TARGET_COLUMN,
    GROUP_COLUMN,
    "max_contact_speed",
    "slip_fraction",
    "time_start",
    "time_end",
    "window_duration",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Required columns missing: {sorted(missing)}"
    )

selected_features = load_selected_features(
    FEATURES_PATH
)

missing_features = (
    set(selected_features) - set(df.columns)
)

if missing_features:
    raise ValueError(
        f"Selected features missing from dataset: "
        f"{sorted(missing_features)[:10]}"
    )

if df[TARGET_COLUMN].isna().any():
    raise ValueError("Missing target labels detected.")

if df[OBJECT_COLUMN].isna().any():
    raise ValueError("Missing object identities detected.")

if df[GROUP_COLUMN].isna().any():
    raise ValueError("Missing trial identifiers detected.")

df[TARGET_COLUMN] = (
    pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise",
    ).astype(np.int8)
)

if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

# Load the original, saved trial split.
with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8",
) as file:
    split_info = json.load(file)

train_trials = set(
    map(str, split_info["train_trials"])
)

test_trials = set(
    map(str, split_info["test_trials"])
)

if train_trials.intersection(test_trials):
    raise ValueError(
        "Trial leakage detected in the saved split."
    )

dataset_trials = set(
    df[GROUP_COLUMN].astype(str).unique()
)

if dataset_trials != train_trials.union(test_trials):
    raise ValueError(
        "Dataset trial identifiers do not match the saved split. "
        "Do not silently create a new split."
    )

print(f"Selected IMU features: {len(selected_features)}")
print(f"Training trials: {len(train_trials)}")
print(f"Testing trials: {len(test_trials)}")
print(f"Objects: {sorted(df[OBJECT_COLUMN].unique())}")


# ============================================================
# 4. PREPARE TRAIN AND TEST DATA
# ============================================================

print("\nPreparing the original train/test split...")

df[GROUP_COLUMN] = df[GROUP_COLUMN].astype(str)

train_mask = df[GROUP_COLUMN].isin(train_trials)
test_mask = df[GROUP_COLUMN].isin(test_trials)

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

if set(train_df[GROUP_COLUMN]).intersection(
    set(test_df[GROUP_COLUMN])
):
    raise ValueError("Train/test trial overlap detected.")

X_train_imu = train_df[selected_features].to_numpy(
    dtype=np.float32
)

X_test_imu = test_df[selected_features].to_numpy(
    dtype=np.float32
)

y_train = train_df[TARGET_COLUMN].to_numpy(
    dtype=np.int8
)

y_test = test_df[TARGET_COLUMN].to_numpy(
    dtype=np.int8
)

groups_train = train_df[GROUP_COLUMN].to_numpy()

object_train = train_df[OBJECT_COLUMN].astype(str).to_numpy()
object_test = test_df[OBJECT_COLUMN].astype(str).to_numpy()

if not np.isfinite(X_train_imu).all():
    raise ValueError("Non-finite training IMU feature values.")

if not np.isfinite(X_test_imu).all():
    raise ValueError("Non-finite test IMU feature values.")

print(f"Training windows: {len(y_train):,}")
print(f"Testing windows: {len(y_test):,}")

print("\nTraining label distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nObject distribution in training set:")
print(train_df[OBJECT_COLUMN].value_counts().sort_index())


# ============================================================
# 5. GROUPED CROSS-VALIDATION FOR OBJECT-AWARE MODEL
# ============================================================

print("\n" + "=" * 78)
print("GROUPED CROSS-VALIDATION: OBJECT-AWARE CATBOOST")
print("=" * 78)

print(
    "Using five-fold GroupKFold, grouping by trial."
)

cv = GroupKFold(n_splits=N_SPLITS)

oof_probabilities = np.full(
    len(y_train),
    np.nan,
    dtype=np.float64,
)

cv_records = []

cv_start = time.time()

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(
        X_train_imu,
        y_train,
        groups=groups_train,
    ),
    start=1,
):
    fit_trials = set(groups_train[fit_idx])
    val_trials = set(groups_train[val_idx])

    if fit_trials.intersection(val_trials):
        raise ValueError(
            f"Trial leakage detected in CV fold {fold}."
        )

    print(
        f"\nFold {fold}/{N_SPLITS}: "
        f"train windows={len(fit_idx):,}, "
        f"validation windows={len(val_idx):,}, "
        f"train trials={len(fit_trials)}, "
        f"validation trials={len(val_trials)}"
    )

    X_fit = pd.DataFrame(
        X_train_imu[fit_idx],
        columns=selected_features,
    )

    X_val = pd.DataFrame(
        X_train_imu[val_idx],
        columns=selected_features,
    )

    # Add object identity as a categorical input.
    X_fit[OBJECT_COLUMN] = object_train[fit_idx]
    X_val[OBJECT_COLUMN] = object_train[val_idx]

    y_fit = y_train[fit_idx]
    y_val = y_train[val_idx]

    model = make_model()

    start = time.time()

    model.fit(
        X_fit,
        y_fit,
        cat_features=[OBJECT_COLUMN],
    )

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    oof_probabilities[val_idx] = val_probabilities

    elapsed = time.time() - start

    for threshold in THRESHOLDS:
        metrics = evaluate_predictions(
            y_val,
            val_probabilities,
            threshold,
        )

        metrics.update({
            "fold": fold,
            "fit_trials": len(fit_trials),
            "validation_trials": len(val_trials),
            "training_seconds": elapsed,
        })

        cv_records.append(metrics)

    print(f"Fold training time: {elapsed:.2f} seconds")

    del model, X_fit, X_val


if np.isnan(oof_probabilities).any():
    raise ValueError(
        "Some training rows did not receive OOF predictions."
    )

cv_elapsed = time.time() - cv_start

cv_df = pd.DataFrame(cv_records)

cv_df.to_csv(
    OUTPUT_DIR / "object_aware_cv_metrics_by_fold.csv",
    index=False,
)

cv_summary = (
    cv_df
    .groupby("threshold")
    .agg({
        "accuracy": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "pr_auc": ["mean", "std"],
    })
)

cv_summary.to_csv(
    OUTPUT_DIR / "object_aware_cv_summary.csv"
)

print(f"\nCV duration: {cv_elapsed / 60:.2f} minutes")

print("\nCross-validation summary:")
print(cv_summary.round(4).to_string())


# ============================================================
# 6. TRAIN FINAL OBJECT-AWARE MODEL
# ============================================================

print("\n" + "=" * 78)
print("TRAINING FINAL OBJECT-AWARE MODEL")
print("=" * 78)

X_train_object = pd.DataFrame(
    X_train_imu,
    columns=selected_features,
)

X_test_object = pd.DataFrame(
    X_test_imu,
    columns=selected_features,
)

X_train_object[OBJECT_COLUMN] = object_train
X_test_object[OBJECT_COLUMN] = object_test

final_model = make_model()

final_start = time.time()

final_model.fit(
    X_train_object,
    y_train,
    cat_features=[OBJECT_COLUMN],
)

final_training_seconds = time.time() - final_start

model_path = (
    MODEL_DIR
    / "catboost_object_identity.cbm"
)

final_model.save_model(str(model_path))

print(f"Saved model: {model_path}")
print(
    f"Final training time: "
    f"{final_training_seconds:.2f} seconds"
)


# ============================================================
# 7. EVALUATE ON THE SAVED TEST SET
# ============================================================

print("\n" + "=" * 78)
print("HELD-OUT TEST EVALUATION")
print("=" * 78)

test_probabilities = final_model.predict_proba(
    X_test_object
)[:, 1]

prediction_records = test_df[
    [
        GROUP_COLUMN,
        OBJECT_COLUMN,
        TARGET_COLUMN,
    ]
].copy()

prediction_records["slip_probability"] = test_probabilities

test_metrics = []

for threshold in THRESHOLDS:
    metrics = evaluate_predictions(
        y_test,
        test_probabilities,
        threshold,
    )

    metrics["model"] = "CatBoost + Object Identity"

    test_metrics.append(metrics)

    print(f"\nThreshold: {threshold:.2f}")

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ):
        print(f"{key:12s}: {metrics[key]:.6f}")

    print(
        "Confusion matrix [[TN, FP], [FN, TP]]:"
    )

    print(
        np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])
    )

    prediction_records[
        f"prediction_threshold_{threshold:.2f}"
    ] = (
        test_probabilities >= threshold
    ).astype(np.int8)


test_metrics_df = pd.DataFrame(test_metrics)

test_metrics_df.to_csv(
    OUTPUT_DIR / "object_aware_test_metrics.csv",
    index=False,
)

prediction_records.to_csv(
    OUTPUT_DIR / "object_aware_test_predictions.csv",
    index=False,
)


# ============================================================
# 8. OBJECT-WISE EVALUATION
# ============================================================

print("\n" + "=" * 78)
print("OBJECT-WISE TEST EVALUATION")
print("=" * 78)

object_records = []

for object_name, object_df in prediction_records.groupby(
    OBJECT_COLUMN
):
    y_object = object_df[TARGET_COLUMN].to_numpy()

    p_object = object_df["slip_probability"].to_numpy()

    print(f"\nObject: {object_name}")

    for threshold in THRESHOLDS:
        metrics = evaluate_predictions(
            y_object,
            p_object,
            threshold,
        )

        metrics.update({
            "object": object_name,
            "windows": len(object_df),
            "actual_slip_windows": int(
                np.sum(y_object == 1)
            ),
            "actual_no_slip_windows": int(
                np.sum(y_object == 0)
            ),
        })

        object_records.append(metrics)

        print(
            f"  Threshold {threshold:.2f}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

object_df = pd.DataFrame(object_records)

object_df.to_csv(
    OUTPUT_DIR / "object_aware_object_wise_metrics.csv",
    index=False,
)


# ============================================================
# 9. SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {
    "experiment": "CatBoost with object identity",
    "selected_imu_features": len(selected_features),
    "object_feature": OBJECT_COLUMN,
    "categorical_features": [OBJECT_COLUMN],
    "training_trials": len(train_trials),
    "testing_trials": len(test_trials),
    "training_windows": len(y_train),
    "testing_windows": len(y_test),
    "cv_splits": N_SPLITS,
    "cv_duration_seconds": cv_elapsed,
    "final_training_seconds": final_training_seconds,
    "thresholds_evaluated": THRESHOLDS,
    "test_metrics": test_metrics,
    "model_path": str(model_path),
    "results_directory": str(OUTPUT_DIR),
    "important_caveat": (
        "Metrics measure agreement with existing derived labels. "
        "The test set has been examined during previous project "
        "experiments and is not a pristine final evaluation set."
    ),
}

save_json(
    summary,
    OUTPUT_DIR / "object_identity_experiment_summary.json",
)

print("\n" + "=" * 78)
print("OBJECT IDENTITY EXPERIMENT COMPLETED")
print("=" * 78)

print("\nResults directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")

print("\nExisting models and datasets were not modified.")