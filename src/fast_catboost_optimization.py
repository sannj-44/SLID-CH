from pathlib import Path
import gc
import json
import time

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


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data_processed" / "slid_ch_features_temporal.csv"
)

SPLIT_PATH = (
    ROOT / "results" / "trial_split.json"
)

FEATURES_PATH = (
    ROOT
    / "results"
    / "catboost_feature_selected"
    / "selected_features.json"
)

OUTPUT_MODEL_DIR = (
    ROOT / "models" / "catboost_fast_optimization"
)

OUTPUT_RESULTS_DIR = (
    ROOT / "results" / "catboost_fast_optimization"
)

OUTPUT_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42
N_SPLITS = 5

# A small, controlled comparison.
CLASS_WEIGHT_CONFIGS = {
    "no_weights": None,
    "balanced": "Balanced",
    "sqrt_balanced": "SqrtBalanced",
}

# Fixed CatBoost parameters from your existing tuning experiment.
BASE_PARAMS = {
    "iterations": 600,
    "depth": 8,
    "learning_rate": 0.1,
    "l2_leaf_reg": 5,
    "random_strength": 0.0,
    "border_count": 64,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "thread_count": -1,
    "allow_writing_files": False,
}

# Thresholds are selected using training OOF predictions only.
THRESHOLDS = np.round(
    np.arange(0.25, 0.701, 0.01),
    2,
)

DEFAULT_THRESHOLD = 0.50


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def make_model(class_weight):
    """Create CatBoost using the specified class-weight setting."""

    params = BASE_PARAMS.copy()

    if class_weight is not None:
        params["auto_class_weights"] = class_weight

    return CatBoostClassifier(**params)


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a chosen threshold."""

    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

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


def calculate_confusion(y_true, probabilities, threshold):
    """Return TN, FP, FN, TP."""

    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(np.int8)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        "threshold": float(threshold),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "matrix": matrix.tolist(),
    }


def print_metrics(title, metrics):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.6f}")


# ============================================================
# 3. VALIDATE FILES
# ============================================================

print("=" * 70)
print("FAST CATBOOST CLASS-WEIGHT OPTIMIZATION")
print("=" * 70)

for path in [
    DATA_PATH,
    SPLIT_PATH,
    FEATURES_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nAll required files exist.")


# ============================================================
# 4. LOAD TRIAL SPLIT
# ============================================================

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8",
) as file:
    split_info = json.load(file)

train_trials = set(
    str(x) for x in split_info["train_trials"]
)

test_trials = set(
    str(x) for x in split_info["test_trials"]
)

if train_trials.intersection(test_trials):
    raise ValueError(
        "Training and test trials overlap."
    )

print(f"\nTraining trials: {len(train_trials)}")
print(f"Testing trials:  {len(test_trials)}")


# ============================================================
# 5. LOAD SELECTED FEATURE LIST
# ============================================================

with open(
    FEATURES_PATH,
    "r",
    encoding="utf-8",
) as file:
    feature_info = json.load(file)

selected_features = feature_info["selected_features"]

expected_count = int(
    feature_info["selected_feature_count"]
)

if len(selected_features) != expected_count:
    raise ValueError(
        "Feature count does not match selected_features.json."
    )

if len(set(selected_features)) != len(selected_features):
    raise ValueError(
        "Duplicate feature names found."
    )

if expected_count != 150:
    raise ValueError(
        f"Expected 150 selected features, found {expected_count}."
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
        "Target-derived or metadata columns appear in the features."
    )

print(f"\nSelected features: {len(selected_features)}")


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\nLoading dataset...")

use_columns = (
    selected_features
    + ["trial", "slip"]
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
    raise ValueError("Dataset is empty.")

if df[selected_features].isna().any().any():
    raise ValueError(
        "Missing values found in selected features."
    )

if not np.isfinite(
    df[selected_features].to_numpy(dtype=np.float32)
).all():
    raise ValueError(
        "Non-finite values found in selected features."
    )

if not set(df["slip"].unique()).issubset({0, 1}):
    raise ValueError(
        "Target must contain only 0 and 1."
    )

dataset_trials = set(
    df["trial"].astype(str).unique()
)

if dataset_trials - train_trials - test_trials:
    raise ValueError(
        "Some trials are absent from the saved trial split."
    )

if train_trials - dataset_trials:
    raise ValueError(
        "Some training trials are missing from the dataset."
    )

if test_trials - dataset_trials:
    raise ValueError(
        "Some test trials are missing from the dataset."
    )

trial_series = df["trial"].astype(str)

train_mask = trial_series.isin(
    train_trials
).to_numpy()

test_mask = trial_series.isin(
    test_trials
).to_numpy()

if not np.all(train_mask | test_mask):
    raise ValueError(
        "Some rows are not assigned to the saved train/test split."
    )

if np.any(train_mask & test_mask):
    raise ValueError(
        "Train/test row overlap detected."
    )

X_all = df[selected_features].to_numpy(
    dtype=np.float32,
    copy=True,
)

y_all = df["slip"].to_numpy(
    dtype=np.int8,
    copy=True,
)

groups_all = trial_series.to_numpy(
    copy=True
)

X_train = X_all[train_mask]
y_train = y_all[train_mask]
groups_train = groups_all[train_mask]

X_test = X_all[test_mask]
y_test = y_all[test_mask]
groups_test = groups_all[test_mask]

del df, X_all, y_all, groups_all, trial_series
gc.collect()

if set(groups_train).intersection(set(groups_test)):
    raise ValueError(
        "Train/test trial leakage detected."
    )

print(f"Training matrix: {X_train.shape}")
print(f"Testing matrix:  {X_test.shape}")

print(
    f"Training classes: No Slip={np.sum(y_train == 0):,}, "
    f"Slip={np.sum(y_train == 1):,}"
)


# ============================================================
# 7. GROUPED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("COMPARING CLASS-WEIGHT CONFIGURATIONS")
print("=" * 70)

cv = GroupKFold(
    n_splits=N_SPLITS
)

configuration_results = []
all_oof_results = []

experiment_start = time.time()

for config_name, class_weight in CLASS_WEIGHT_CONFIGS.items():

    print("\n" + "#" * 70)
    print(f"CONFIGURATION: {config_name}")
    print("#" * 70)

    oof_probabilities = np.full(
        len(y_train),
        np.nan,
        dtype=np.float64,
    )

    fold_records = []

    config_start = time.time()

    for fold_number, (fit_idx, val_idx) in enumerate(
        cv.split(
            X_train,
            y_train,
            groups_train,
        ),
        start=1,
    ):

        fit_groups = set(
            groups_train[fit_idx]
        )

        val_groups = set(
            groups_train[val_idx]
        )

        if fit_groups.intersection(val_groups):
            raise ValueError(
                f"Trial leakage in fold {fold_number}."
            )

        print(
            f"\n{config_name}: "
            f"fold {fold_number}/{N_SPLITS}"
        )

        fold_model = make_model(
            class_weight
        )

        fold_start = time.time()

        fold_model.fit(
            X_train[fit_idx],
            y_train[fit_idx],
        )

        probabilities = fold_model.predict_proba(
            X_train[val_idx]
        )[:, 1]

        # Preserve original training-array row positions.
        oof_probabilities[val_idx] = probabilities

        fold_metrics = calculate_metrics(
            y_train[val_idx],
            probabilities,
            DEFAULT_THRESHOLD,
        )

        fold_metrics["fold"] = fold_number
        fold_records.append(fold_metrics)

        print(
            f"Accuracy={fold_metrics['accuracy']:.4f} | "
            f"F1={fold_metrics['f1_score']:.4f} | "
            f"Time={time.time() - fold_start:.1f}s"
        )

        del fold_model, probabilities
        gc.collect()

    if not np.isfinite(oof_probabilities).all():
        raise RuntimeError(
            f"Incomplete OOF predictions for {config_name}."
        )

    # Evaluate the full aligned OOF predictions.
    oof_metrics = calculate_metrics(
        y_train,
        oof_probabilities,
        DEFAULT_THRESHOLD,
    )

    record = {
        "configuration": config_name,
        "class_weight": (
            class_weight
            if class_weight is not None
            else "None"
        ),
        "cv_accuracy_at_050": oof_metrics["accuracy"],
        "cv_precision_at_050": oof_metrics["precision"],
        "cv_recall_at_050": oof_metrics["recall"],
        "cv_f1_at_050": oof_metrics["f1_score"],
        "cv_roc_auc": oof_metrics["roc_auc"],
        "cv_pr_auc": oof_metrics["pr_auc"],
        "runtime_seconds": round(
            time.time() - config_start,
            2,
        ),
    }

    configuration_results.append(record)

    print_metrics(
        f"{config_name}: FULL OOF METRICS AT THRESHOLD 0.50",
        oof_metrics,
    )

    for i in range(len(y_train)):
        all_oof_results.append({
            "configuration": config_name,
            "train_row_index": i,
            "trial": groups_train[i],
            "actual_slip": int(y_train[i]),
            "oof_probability": float(
                oof_probabilities[i]
            ),
        })

    del oof_probabilities
    gc.collect()


# ============================================================
# 8. SELECT CONFIGURATION USING OOF F1
# ============================================================

comparison_df = pd.DataFrame(
    configuration_results
)

comparison_df = comparison_df.sort_values(
    [
        "cv_f1_at_050",
        "cv_accuracy_at_050",
    ],
    ascending=False,
).reset_index(drop=True)

comparison_path = (
    OUTPUT_RESULTS_DIR
    / "class_weight_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)

print("\n" + "=" * 70)
print("CLASS-WEIGHT COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

best_configuration = comparison_df.iloc[0]

best_config_name = str(
    best_configuration["configuration"]
)

best_class_weight = (
    None
    if best_configuration["class_weight"] == "None"
    else str(best_configuration["class_weight"])
)

print(
    f"\nSelected configuration by OOF F1: "
    f"{best_config_name}"
)


# ============================================================
# 9. REGENERATE OOF PROBABILITIES FOR THE WINNER
#    AND OPTIMIZE THE THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZING THE WINNING CONFIGURATION'S THRESHOLD")
print("=" * 70)

best_oof_probabilities = np.full(
    len(y_train),
    np.nan,
    dtype=np.float64,
)

for fold_number, (fit_idx, val_idx) in enumerate(
    cv.split(
        X_train,
        y_train,
        groups_train,
    ),
    start=1,
):

    print(
        f"\nWinner OOF fold {fold_number}/{N_SPLITS}"
    )

    fold_model = make_model(
        best_class_weight
    )

    fold_model.fit(
        X_train[fit_idx],
        y_train[fit_idx],
    )

    best_oof_probabilities[val_idx] = (
        fold_model.predict_proba(
            X_train[val_idx]
        )[:, 1]
    )

    del fold_model
    gc.collect()

if not np.isfinite(best_oof_probabilities).all():
    raise RuntimeError(
        "Incomplete winner OOF predictions."
    )

threshold_records = []

for threshold in THRESHOLDS:

    metrics = calculate_metrics(
        y_train,
        best_oof_probabilities,
        float(threshold),
    )

    threshold_records.append(metrics)

threshold_df = pd.DataFrame(
    threshold_records
)

threshold_df.to_csv(
    OUTPUT_RESULTS_DIR / "threshold_comparison.csv",
    index=False,
)

max_f1 = threshold_df["f1_score"].max()

candidates = threshold_df[
    np.isclose(
        threshold_df["f1_score"],
        max_f1,
        atol=1e-12,
        rtol=0,
    )
].copy()

candidates["distance_from_050"] = (
    candidates["threshold"] - DEFAULT_THRESHOLD
).abs()

candidates = candidates.sort_values(
    [
        "distance_from_050",
        "threshold",
    ],
    ascending=[True, True],
)

best_threshold = float(
    candidates.iloc[0]["threshold"]
)

print(
    f"\nSelected threshold from training OOF F1: "
    f"{best_threshold:.2f}"
)

winner_oof_metrics_050 = calculate_metrics(
    y_train,
    best_oof_probabilities,
    DEFAULT_THRESHOLD,
)

winner_oof_metrics_best = calculate_metrics(
    y_train,
    best_oof_probabilities,
    best_threshold,
)

print_metrics(
    "WINNING CONFIGURATION: OOF METRICS AT 0.50",
    winner_oof_metrics_050,
)

print_metrics(
    f"WINNING CONFIGURATION: OOF METRICS AT {best_threshold:.2f}",
    winner_oof_metrics_best,
)

pd.DataFrame({
    "trial": groups_train,
    "actual_slip": y_train,
    "oof_probability": best_oof_probabilities,
}).to_csv(
    OUTPUT_RESULTS_DIR / "winner_oof_predictions.csv",
    index=False,
)


# ============================================================
# 10. TRAIN WINNER ON ALL TRAINING TRIALS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL CATBOOST MODEL")
print("=" * 70)

final_model = make_model(
    best_class_weight
)

training_start = time.time()

final_model.fit(
    X_train,
    y_train,
)

final_training_seconds = (
    time.time() - training_start
)

model_path = (
    OUTPUT_MODEL_DIR
    / "catboost_fast_best.cbm"
)

final_model.save_model(
    str(model_path)
)

print(f"\nSaved model:\n{model_path}")


# ============================================================
# 11. EVALUATE ON HELD-OUT TEST SET
# ============================================================

print("\n" + "=" * 70)
print("FINAL HELD-OUT TEST EVALUATION")
print("=" * 70)

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

test_metrics_050 = calculate_metrics(
    y_test,
    test_probabilities,
    DEFAULT_THRESHOLD,
)

test_metrics_best = calculate_metrics(
    y_test,
    test_probabilities,
    best_threshold,
)

print_metrics(
    "TEST RESULTS AT THRESHOLD 0.50",
    test_metrics_050,
)

print_metrics(
    f"TEST RESULTS AT THRESHOLD {best_threshold:.2f}",
    test_metrics_best,
)

test_predictions_050 = (
    test_probabilities >= DEFAULT_THRESHOLD
).astype(np.int8)

test_predictions_best = (
    test_probabilities >= best_threshold
).astype(np.int8)

cm_050 = confusion_matrix(
    y_test,
    test_predictions_050,
    labels=[0, 1],
)

cm_best = confusion_matrix(
    y_test,
    test_predictions_best,
    labels=[0, 1],
)

print("\nConfusion matrix at threshold 0.50:")
print(cm_050)

print(
    f"\nConfusion matrix at threshold {best_threshold:.2f}:"
)
print(cm_best)


# ============================================================
# 12. SAVE TEST RESULTS
# ============================================================

test_results = pd.DataFrame([
    {
        "configuration": best_config_name,
        **test_metrics_050,
    },
    {
        "configuration": best_config_name,
        **test_metrics_best,
    },
])

test_results.to_csv(
    OUTPUT_RESULTS_DIR / "test_metrics.csv",
    index=False,
)

pd.DataFrame({
    "trial": groups_test,
    "actual_slip": y_test,
    "slip_probability": test_probabilities,
    "prediction_threshold_050": test_predictions_050,
    "prediction_optimized_threshold": test_predictions_best,
}).to_csv(
    OUTPUT_RESULTS_DIR / "test_predictions.csv",
    index=False,
)

with open(
    OUTPUT_RESULTS_DIR / "test_confusion_matrices.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "threshold_050": {
                "matrix": cm_050.tolist(),
                "TN": int(cm_050[0, 0]),
                "FP": int(cm_050[0, 1]),
                "FN": int(cm_050[1, 0]),
                "TP": int(cm_050[1, 1]),
            },
            "optimized_threshold": {
                "threshold": best_threshold,
                "matrix": cm_best.tolist(),
                "TN": int(cm_best[0, 0]),
                "FP": int(cm_best[0, 1]),
                "FN": int(cm_best[1, 0]),
                "TP": int(cm_best[1, 1]),
            },
        },
        file,
        indent=4,
    )


# ============================================================
# 13. SAVE CONFIGURATION AND SUMMARY
# ============================================================

summary = {
    "experiment": "Fast CatBoost class-weight optimization",
    "selected_configuration": best_config_name,
    "class_weight": (
        best_class_weight
        if best_class_weight is not None
        else "None"
    ),
    "selected_feature_count": len(selected_features),
    "training_trials": len(train_trials),
    "testing_trials": len(test_trials),
    "training_windows": len(y_train),
    "testing_windows": len(y_test),
    "cv_method": "GroupKFold",
    "cv_folds": N_SPLITS,
    "selection_metric": "OOF F1-score at threshold 0.50",
    "selected_threshold": best_threshold,
    "oof_metrics_at_050": winner_oof_metrics_050,
    "oof_metrics_at_selected_threshold": winner_oof_metrics_best,
    "test_metrics_at_050": test_metrics_050,
    "test_metrics_at_selected_threshold": test_metrics_best,
    "final_training_seconds": round(
        final_training_seconds,
        2,
    ),
    "model_path": str(model_path),
    "class_weight_comparison_path": str(comparison_path),
}

with open(
    OUTPUT_RESULTS_DIR / "summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FAST CATBOOST OPTIMIZATION COMPLETED")
print("=" * 70)

print(f"Winning configuration: {best_config_name}")
print(f"Class weight: {best_class_weight}")
print(f"Selected threshold: {best_threshold:.2f}")

print("\nTEST COMPARISON")
print("-" * 70)

print(
    f"{'Metric':<15}"
    f"{'Threshold 0.50':>20}"
    f"{'Optimized':>20}"
)

print("-" * 70)

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]:
    print(
        f"{metric:<15}"
        f"{test_metrics_050[metric]:>20.6f}"
        f"{test_metrics_best[metric]:>20.6f}"
    )

print("-" * 70)

print(
    f"\nModel directory:\n{OUTPUT_MODEL_DIR}"
)

print(
    f"\nResults directory:\n{OUTPUT_RESULTS_DIR}"
)

print("\nExisting models and results were not overwritten.")
print("Done.")