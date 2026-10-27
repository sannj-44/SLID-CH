from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd

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

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data_processed" / "slid_ch_features_temporal.csv"
)

SPLIT_PATH = ROOT / "results" / "trial_split.json"

CATBOOST_MODEL_PATH = (
    ROOT
    / "models"
    / "catboost_feature_selected"
    / "catboost_selected_150_features.cbm"
)

SELECTED_FEATURES_PATH = (
    ROOT
    / "results"
    / "catboost_feature_selected"
    / "selected_features.json"
)

XGBOOST_MODEL_PATH = (
    ROOT
    / "models"
    / "xgboost_tuned"
    / "xgboost_tuned_model.joblib"
)

RESULT_DIR = ROOT / "results" / "ensemble_soft_voting"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

THRESHOLD = 0.5


# ============================================================
# 2. VALIDATE FILES
# ============================================================

print("=" * 75)
print("SOFT-VOTING ENSEMBLE EXPERIMENT")
print("=" * 75)

required_files = [
    DATA_PATH,
    SPLIT_PATH,
    CATBOOST_MODEL_PATH,
    SELECTED_FEATURES_PATH,
    XGBOOST_MODEL_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

print("\nAll required files exist.")


# ============================================================
# 3. LOAD THE SAVED TRIAL SPLIT
# ============================================================

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8",
) as file:
    split_info = json.load(file)

train_trials = set(
    split_info["train_trials"]
)

test_trials = set(
    split_info["test_trials"]
)

if train_trials.intersection(test_trials):
    raise ValueError(
        "Train/test trial overlap detected."
    )

print(f"\nTraining trials: {len(train_trials)}")
print(f"Testing trials:  {len(test_trials)}")


# ============================================================
# 4. IDENTIFY ORIGINAL 72 FEATURES
# ============================================================

print("\nReading dataset column names...")

column_names = pd.read_csv(
    DATA_PATH,
    nrows=0,
).columns.tolist()

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

if len(original_features) != 72:
    raise ValueError(
        f"Expected 72 original IMU features; "
        f"found {len(original_features)}."
    )

print(
    f"Original IMU features: {len(original_features)}"
)


# ============================================================
# 5. LOAD CATBOOST SELECTED FEATURES
# ============================================================

with open(
    SELECTED_FEATURES_PATH,
    "r",
    encoding="utf-8",
) as file:
    selected_info = json.load(file)

selected_features = selected_info[
    "selected_features"
]

if len(selected_features) != 150:
    raise ValueError(
        f"Expected 150 selected features; "
        f"found {len(selected_features)}."
    )

if len(set(selected_features)) != len(selected_features):
    raise ValueError(
        "Duplicate features in CatBoost selected-feature list."
    )

missing_selected_features = (
    set(selected_features) - set(column_names)
)

if missing_selected_features:
    raise ValueError(
        "Selected features are missing from the dataset: "
        f"{list(missing_selected_features)[:10]}"
    )

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

if forbidden_features.intersection(selected_features):
    raise ValueError(
        "Target, metadata, or label-derived features "
        "found in the selected CatBoost features."
    )

print(
    f"CatBoost selected features: {len(selected_features)}"
)


# ============================================================
# 6. LOAD TRAINED MODELS
# ============================================================

print("\nLoading trained models...")

catboost_model = CatBoostClassifier()

catboost_model.load_model(
    str(CATBOOST_MODEL_PATH)
)

xgboost_model = joblib.load(
    XGBOOST_MODEL_PATH
)

print("CatBoost model loaded.")
print("XGBoost model loaded.")


# ============================================================
# 7. VALIDATE XGBOOST INPUT DIMENSIONS
# ============================================================

# The original tuned XGBoost model was trained using 72
# statistical IMU features.

if hasattr(xgboost_model, "n_features_in_"):
    xgb_feature_count = int(
        xgboost_model.n_features_in_
    )

    if xgb_feature_count != 72:
        raise ValueError(
            f"XGBoost expects {xgb_feature_count} features, "
            "but this script expects the original 72."
        )

print("XGBoost input feature count validated.")


# ============================================================
# 8. LOAD ONLY REQUIRED DATA
# ============================================================

print("\nLoading dataset...")

# The models need the union of:
# - 72 original IMU features for XGBoost
# - 150 selected features for CatBoost
# - trial and slip columns for splitting and evaluation

required_features = list(
    dict.fromkeys(
        original_features + selected_features
    )
)

use_columns = (
    required_features
    + ["trial", "slip"]
)

dtype_map = {
    column: np.float32
    for column in required_features
}

dtype_map["trial"] = str
dtype_map["slip"] = np.int8

df = pd.read_csv(
    DATA_PATH,
    usecols=use_columns,
    dtype=dtype_map,
)

print(f"Rows loaded: {len(df):,}")

if df["trial"].isna().any():
    raise ValueError(
        "Missing trial identifiers detected."
    )

if df["slip"].isna().any():
    raise ValueError(
        "Missing target labels detected."
    )

dataset_trials = set(
    df["trial"].unique()
)

unassigned_trials = (
    dataset_trials - train_trials - test_trials
)

if unassigned_trials:
    raise ValueError(
        "Some trials are missing from the saved split: "
        f"{list(unassigned_trials)[:10]}"
    )

train_mask = df["trial"].isin(
    train_trials
).to_numpy()

test_mask = df["trial"].isin(
    test_trials
).to_numpy()

if np.any(train_mask & test_mask):
    raise ValueError(
        "A row belongs to both train and test."
    )

if not np.all(train_mask | test_mask):
    raise ValueError(
        "Some rows are not assigned to train or test."
    )

# Keep test windows in their original dataset order.
y_all = df["slip"].to_numpy(
    dtype=np.int8,
    copy=True,
)

y_test = y_all[test_mask]

X_catboost_all = df[selected_features].to_numpy(
    dtype=np.float32,
    copy=True,
)

X_xgboost_all = df[original_features].to_numpy(
    dtype=np.float32,
    copy=True,
)

del df
gc.collect()

X_catboost_test = X_catboost_all[test_mask]

X_xgboost_test = X_xgboost_all[test_mask]

del X_catboost_all, X_xgboost_all, y_all
gc.collect()

if not np.isfinite(X_catboost_test).all():
    raise ValueError(
        "CatBoost test features contain NaN or infinity."
    )

if not np.isfinite(X_xgboost_test).all():
    raise ValueError(
        "XGBoost test features contain NaN or infinity."
    )

print("\nTest data prepared.")
print(
    f"Test windows: {len(y_test):,}"
)
print(
    f"CatBoost test shape: {X_catboost_test.shape}"
)
print(
    f"XGBoost test shape:  {X_xgboost_test.shape}"
)


# ============================================================
# 9. GENERATE INDIVIDUAL MODEL PROBABILITIES
# ============================================================

print("\nGenerating CatBoost predictions...")

start_time = time.time()

catboost_probabilities = (
    catboost_model.predict_proba(
        X_catboost_test
    )[:, 1]
)

catboost_prediction_time = (
    time.time() - start_time
)

print(
    f"CatBoost prediction time: "
    f"{catboost_prediction_time:.2f} seconds"
)

print("\nGenerating XGBoost predictions...")

start_time = time.time()

xgboost_probabilities = (
    xgboost_model.predict_proba(
        X_xgboost_test
    )[:, 1]
)

xgboost_prediction_time = (
    time.time() - start_time
)

print(
    f"XGBoost prediction time: "
    f"{xgboost_prediction_time:.2f} seconds"
)


# ============================================================
# 10. CREATE THE SOFT-VOTING ENSEMBLE
# ============================================================

# Equal-weight soft voting:
#
# P(ensemble) =
#     0.5 * P(CatBoost)
#   + 0.5 * P(XGBoost)

ensemble_probabilities = (
    0.5 * catboost_probabilities
    + 0.5 * xgboost_probabilities
)

ensemble_predictions = (
    ensemble_probabilities >= THRESHOLD
).astype(np.int8)


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    probabilities,
    threshold=0.5,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
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


# ============================================================
# 12. EVALUATE ALL THREE MODELS
# ============================================================

catboost_metrics = evaluate_model(
    "Selected CatBoost",
    y_test,
    catboost_probabilities,
    THRESHOLD,
)

xgboost_metrics = evaluate_model(
    "Tuned XGBoost",
    y_test,
    xgboost_probabilities,
    THRESHOLD,
)

ensemble_metrics = evaluate_model(
    "Equal-Weight Soft Voting",
    y_test,
    ensemble_probabilities,
    THRESHOLD,
)

comparison_df = pd.DataFrame([
    catboost_metrics,
    xgboost_metrics,
    ensemble_metrics,
])

comparison_path = (
    RESULT_DIR / "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

display_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]

print(
    comparison_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}",
    )
)


# ============================================================
# 13. ENSEMBLE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    ensemble_predictions,
    labels=[0, 1],
)

print("\nEnsemble confusion matrix:")
print(cm)

report = classification_report(
    y_test,
    ensemble_predictions,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    zero_division=0,
)

print("\nEnsemble classification report:")
print(report)

with open(
    RESULT_DIR / "ensemble_classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

pd.DataFrame(
    cm,
    index=[
        "Actual No Slip",
        "Actual Slip",
    ],
    columns=[
        "Predicted No Slip",
        "Predicted Slip",
    ],
).to_csv(
    RESULT_DIR / "ensemble_confusion_matrix.csv"
)


# ============================================================
# 14. SAVE ALL TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "actual_slip": y_test,
    "catboost_probability": catboost_probabilities,
    "xgboost_probability": xgboost_probabilities,
    "ensemble_probability": ensemble_probabilities,
    "catboost_prediction": (
        catboost_probabilities >= THRESHOLD
    ).astype(np.int8),
    "xgboost_prediction": (
        xgboost_probabilities >= THRESHOLD
    ).astype(np.int8),
    "ensemble_prediction": ensemble_predictions,
})

predictions_df.to_csv(
    RESULT_DIR / "ensemble_test_predictions.csv",
    index=False,
)


# ============================================================
# 15. SAVE METRICS AND EXPERIMENT CONFIGURATION
# ============================================================

experiment_summary = {
    "experiment": "Equal-weight soft-voting ensemble",
    "test_windows": int(len(y_test)),
    "training_trials": int(len(train_trials)),
    "testing_trials": int(len(test_trials)),
    "threshold": THRESHOLD,
    "weights": {
        "selected_catboost": 0.5,
        "tuned_xgboost": 0.5,
    },
    "catboost_features": len(selected_features),
    "xgboost_features": len(original_features),
    "catboost_prediction_seconds": round(
        catboost_prediction_time, 4
    ),
    "xgboost_prediction_seconds": round(
        xgboost_prediction_time, 4
    ),
    "results": {
        "selected_catboost": catboost_metrics,
        "tuned_xgboost": xgboost_metrics,
        "soft_voting_ensemble": ensemble_metrics,
    },
}

with open(
    RESULT_DIR / "ensemble_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        experiment_summary,
        file,
        indent=4,
    )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SOFT-VOTING EXPERIMENT COMPLETED")
print("=" * 75)

print("\nEqual-weight ensemble results:")

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]:
    print(
        f"{metric}: "
        f"{ensemble_metrics[metric]:.4%}"
    )

print(f"\nResults directory: {RESULT_DIR}")
print(
    f"Model comparison: {comparison_path}"
)

print(
    "\nNote: This script evaluates the ensemble but "
    "does not save a new standalone ensemble model."
)