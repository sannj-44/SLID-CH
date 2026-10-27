"""
CatBoost Hyperparameter Tuning for SLID-CH Slip Detection
=========================================================

Objectives:
1. Load the existing processed SLID-CH feature dataset.
2. Reuse the existing trial-level train/test split.
3. Perform trial-grouped cross-validation on training trials.
4. Tune CatBoost hyperparameters using randomized search.
5. Evaluate the selected model on the untouched test trials.
6. Save the tuned model, parameters, metrics, reports and plots.

Important:
- Original dataset files are never modified.
- Existing models and results are never overwritten.
- Label-derived columns are excluded from model inputs.
- Train/test separation is maintained at the trial level.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier

from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)

from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data_processed" / "slid_ch_features.csv"
SPLIT_FILE = PROJECT_ROOT / "results" / "trial_split.json"

MODEL_DIR = PROJECT_ROOT / "models" / "catboost_tuned"
RESULT_DIR = PROJECT_ROOT / "results" / "catboost_tuned"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Five-fold cross-validation grouped by trial.
N_SPLITS = 5

# Initial randomized search size.
# Increase this later if the initial search is promising.
N_ITER = 12

# Parallel search jobs.
# Each CatBoost model also uses multiple threads, so keep this
# at 1 initially to avoid CPU/RAM oversubscription.
N_JOBS = 1

# CatBoost training threads.
THREAD_COUNT = -1

# Optimize accuracy during hyperparameter selection.
SCORING = "accuracy"

# Preserve the same probability decision threshold as the baseline.
DECISION_THRESHOLD = 0.5


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("CATBOOST HYPERPARAMETER TUNING")
print("=" * 75)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_FILE}"
    )

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Existing trial split not found:\n{SPLIT_FILE}\n"
        "Run your existing model training pipeline first."
    )

print(f"\nLoading dataset:\n{DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")

with open(SPLIT_FILE, "r", encoding="utf-8") as file:
    split_info = json.load(file)

train_trials = set(split_info["train_trials"])
test_trials = set(split_info["test_trials"])

if train_trials.intersection(test_trials):
    raise RuntimeError(
        "Trial leakage detected in the saved split."
    )

print(f"Training trials in saved split: {len(train_trials)}")
print(f"Testing trials in saved split:  {len(test_trials)}")


# ============================================================
# 4. SELECT IMU FEATURES
# ============================================================

# Only the original 72 IMU statistical features are used.
#
# Excluded:
# - time_start
# - time_end
# - window_duration
# - max_contact_speed
# - slip_fraction
# - object
# - trial
# - slip (target)

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

if len(feature_columns) != 72:
    raise ValueError(
        f"Expected 72 IMU features, found {len(feature_columns)}. "
        "Inspect the processed dataset before proceeding."
    )

required_columns = {"trial", "slip"}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="raise",
)

y = pd.to_numeric(
    df["slip"],
    errors="raise",
).astype(int)

groups = df["trial"].astype(str)

if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
    raise ValueError(
        "IMU features contain NaN or infinite values."
    )

if not set(y.unique()).issubset({0, 1}):
    raise ValueError(
        "Target column must contain only binary labels 0 and 1."
    )

if groups.isna().any():
    raise ValueError(
        "Trial identifiers contain missing values."
    )

print(f"\nNumber of IMU features: {len(feature_columns)}")
print(f"Total windows: {len(df):,}")


# ============================================================
# 5. RECONSTRUCT THE EXISTING TRIAL SPLIT
# ============================================================

# The existing split file records the trial names, not the
# individual window indices. Reconstruct the split using trial IDs.

train_mask = groups.isin(train_trials)
test_mask = groups.isin(test_trials)

# Every dataset row must belong to exactly one of the two splits.
unassigned_mask = ~(train_mask | test_mask)

if unassigned_mask.any():
    missing_trials = sorted(
        groups.loc[unassigned_mask].unique().tolist()
    )

    raise ValueError(
        "Some dataset trials are absent from the saved split:\n"
        f"{missing_trials[:20]}"
    )

X_train = X.loc[train_mask].reset_index(drop=True)
y_train = y.loc[train_mask].reset_index(drop=True)
groups_train = groups.loc[train_mask].reset_index(drop=True)

X_test = X.loc[test_mask].reset_index(drop=True)
y_test = y.loc[test_mask].reset_index(drop=True)
groups_test = groups.loc[test_mask].reset_index(drop=True)

if set(groups_train).intersection(set(groups_test)):
    raise RuntimeError(
        "Train/test trial overlap detected."
    )

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Both train and test sets must contain both classes."
    )

print("\n" + "-" * 75)
print("EXISTING TRAIN/TEST SPLIT")
print("-" * 75)

print(f"Training windows: {len(X_train):,}")
print(f"Testing windows:  {len(X_test):,}")
print(f"Training trials:  {groups_train.nunique()}")
print(f"Testing trials:   {groups_test.nunique()}")
print(
    "Shared trials:    "
    f"{len(set(groups_train).intersection(set(groups_test)))}"
)

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 6. BUILD TRIAL-GROUPED CROSS-VALIDATION
# ============================================================

# Each trial must belong entirely to one fold.
# This prevents overlapping windows from the same trial from
# appearing in both the training and validation portions.

if groups_train.nunique() < N_SPLITS:
    raise ValueError(
        f"Need at least {N_SPLITS} training trials for GroupKFold."
    )

cv = GroupKFold(n_splits=N_SPLITS)

cv_splits = list(
    cv.split(
        X_train,
        y_train,
        groups=groups_train,
    )
)

for fold_number, (fold_train_idx, fold_valid_idx) in enumerate(
    cv_splits,
    start=1,
):
    fold_train_trials = set(
        groups_train.iloc[fold_train_idx]
    )

    fold_valid_trials = set(
        groups_train.iloc[fold_valid_idx]
    )

    if fold_train_trials.intersection(fold_valid_trials):
        raise RuntimeError(
            f"Trial leakage detected in CV fold {fold_number}."
        )

    if (
        y_train.iloc[fold_train_idx].nunique() != 2
        or y_train.iloc[fold_valid_idx].nunique() != 2
    ):
        raise ValueError(
            f"Fold {fold_number} does not contain both classes. "
            "Inspect class distribution by trial and revise the "
            "cross-validation strategy if necessary."
        )

    print(
        f"Fold {fold_number}: "
        f"{len(fold_train_trials)} training trials, "
        f"{len(fold_valid_trials)} validation trials"
    )


# ============================================================
# 7. HYPERPARAMETER SEARCH SPACE
# ============================================================

# Start with a manageable search.
# Expand N_ITER or the parameter distributions later if needed.

param_distributions = {
    "iterations": [400, 600, 800, 1000],
    "depth": [5, 6, 7, 8, 9, 10],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "l2_leaf_reg": [3, 5, 7, 9],
    "random_strength": [0.0, 0.5, 1.0, 2.0],
    "border_count": [64, 128, 254],
}


# ============================================================
# 8. INITIALIZE CATBOOST
# ============================================================

# The original CatBoost model used automatic class weights.
# We preserve that choice in this first tuning experiment so
# hyperparameter tuning is the main variable being investigated.

base_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Logloss",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    thread_count=THREAD_COUNT,
    verbose=False,
    allow_writing_files=False,
)


# ============================================================
# 9. RANDOMIZED SEARCH
# ============================================================

print("\n" + "=" * 75)
print("STARTING RANDOMIZED SEARCH")
print("=" * 75)

print(f"Search iterations: {N_ITER}")
print(f"Cross-validation folds: {N_SPLITS}")
print(f"Scoring metric: {SCORING}")
print(
    f"Approximate model fits: "
    f"{N_ITER * N_SPLITS}"
)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring=SCORING,
    cv=cv_splits,
    refit=True,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=2,
    return_train_score=True,
    error_score="raise",
)

search_start = time.perf_counter()

search.fit(
    X_train,
    y_train,
)

search_time = time.perf_counter() - search_start

print(
    f"\nRandomized search completed in "
    f"{search_time / 60:.2f} minutes."
)


# ============================================================
# 10. SAVE CROSS-VALIDATION RESULTS
# ============================================================

cv_results = pd.DataFrame(search.cv_results_)

cv_results = cv_results.sort_values(
    by="rank_test_score"
)

cv_results_path = (
    RESULT_DIR / "cross_validation_results.csv"
)

cv_results.to_csv(
    cv_results_path,
    index=False,
)

print(f"\nCV results saved to:\n{cv_results_path}")

print("\nTop configurations:")

display_columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "mean_fit_time",
    "params",
]

print(
    cv_results[display_columns]
    .head(5)
    .to_string(index=False)
)


# ============================================================
# 11. SAVE BEST PARAMETERS
# ============================================================

best_parameters = {
    key: (
        value.item()
        if isinstance(value, np.generic)
        else value
    )
    for key, value in search.best_params_.items()
}

best_parameters_path = (
    RESULT_DIR / "best_parameters.json"
)

with open(
    best_parameters_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        best_parameters,
        file,
        indent=4,
    )

print("\nBest parameters:")
print(
    json.dumps(
        best_parameters,
        indent=4,
    )
)

print(
    f"\nBest mean cross-validation accuracy: "
    f"{search.best_score_:.4%}"
)


# ============================================================
# 12. SAVE TUNED MODEL
# ============================================================

best_model = search.best_estimator_

model_path = (
    MODEL_DIR / "catboost_tuned_model.cbm"
)

best_model.save_model(
    str(model_path)
)

print(f"\nTuned model saved to:\n{model_path}")


# ============================================================
# 13. EVALUATE ON THE UNTOUCHED TEST SET
# ============================================================

print("\n" + "=" * 75)
print("FINAL HELD-OUT TEST EVALUATION")
print("=" * 75)

print(
    "The test trials were not used for hyperparameter selection."
)

test_start = time.perf_counter()

y_probability = best_model.predict_proba(
    X_test
)[:, 1]

prediction_time = time.perf_counter() - test_start

y_pred = (
    y_probability >= DECISION_THRESHOLD
).astype(int)

metrics = {
    "model": "CatBoost Tuned",
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "precision": float(
        precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )
    ),
    "f1_score": float(
        f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_test,
            y_probability,
        )
    ),
    "pr_auc": float(
        average_precision_score(
            y_test,
            y_probability,
        )
    ),
    "decision_threshold": DECISION_THRESHOLD,
    "best_cv_accuracy": float(search.best_score_),
    "search_time_seconds": float(search_time),
    "prediction_time_seconds": float(prediction_time),
    "train_windows": int(len(X_train)),
    "test_windows": int(len(X_test)),
    "train_trials": int(groups_train.nunique()),
    "test_trials": int(groups_test.nunique()),
    "number_of_features": int(len(feature_columns)),
}

metrics_path = RESULT_DIR / "metrics.json"

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=4,
    )

print("\nFinal test metrics:")

for metric_name, metric_value in metrics.items():
    if isinstance(metric_value, float):
        if metric_name.endswith("_seconds"):
            print(f"{metric_name}: {metric_value:.2f}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
    else:
        print(f"{metric_name}: {metric_value}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    digits=4,
    zero_division=0,
)

print("\nClassification report:")
print(report)

report_path = (
    RESULT_DIR / "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:
    file.write(report)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
)

print("\nConfusion matrix:")
print(cm)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Slip", "Slip"],
    yticklabels=["No Slip", "Slip"],
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("CatBoost Tuned - Confusion Matrix")

plt.tight_layout()

confusion_matrix_path = (
    RESULT_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

importance_values = best_model.get_feature_importance()

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance_values,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

importance_path = (
    RESULT_DIR / "feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False,
)

top_features = importance_df.head(20).sort_values(
    "importance",
    ascending=True,
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"],
    top_features["importance"],
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("CatBoost Tuned - Top 20 Features")

plt.tight_layout()

feature_plot_path = (
    RESULT_DIR / "feature_importance.png"
)

plt.savefig(
    feature_plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 17. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("CATBOOST TUNING COMPLETED")
print("=" * 75)

print(f"Best CV accuracy: {search.best_score_:.4%}")
print(f"Test accuracy:    {metrics['accuracy']:.4%}")
print(f"Test F1 score:    {metrics['f1_score']:.4%}")
print(f"Test ROC-AUC:     {metrics['roc_auc']:.4%}")

print("\nSaved outputs:")
print(f"  Model:              {model_path}")
print(f"  Best parameters:    {best_parameters_path}")
print(f"  CV results:         {cv_results_path}")
print(f"  Metrics:            {metrics_path}")
print(f"  Classification:     {report_path}")
print(f"  Confusion matrix:   {confusion_matrix_path}")
print(f"  Feature importance: {importance_path}")
print(f"  Feature plot:       {feature_plot_path}")

print("\nOriginal dataset and baseline model were not modified.")
print("=" * 75)