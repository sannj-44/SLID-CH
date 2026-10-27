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
    ROOT / "data_processed" / "slid_ch_features_temporal.csv"
)

SPLIT_PATH = ROOT / "results" / "trial_split.json"

IMPORTANCE_PATH = (
    ROOT / "results" / "catboost_temporal" / "feature_importance.csv"
)

MODEL_DIR = ROOT / "models" / "catboost_feature_selected"
RESULT_DIR = ROOT / "results" / "catboost_feature_selected"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
THRESHOLD = 0.5
N_SPLITS = 5

FEATURE_COUNTS = [288, 200, 150, 100, 72]

# Tuned CatBoost parameters from the previous experiment.
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
# 2. VALIDATE INPUT FILES
# ============================================================

print("=" * 75)
print("CATBOOST FEATURE SELECTION")
print("=" * 75)

for path in [DATA_PATH, SPLIT_PATH, IMPORTANCE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

with open(SPLIT_PATH, "r", encoding="utf-8") as file:
    split_info = json.load(file)

train_trials = set(split_info["train_trials"])
test_trials = set(split_info["test_trials"])

if train_trials.intersection(test_trials):
    raise ValueError("Train/test trial overlap detected.")

print(f"Training trials: {len(train_trials)}")
print(f"Testing trials:  {len(test_trials)}")

# ============================================================
# 3. IDENTIFY THE 288 MODEL FEATURES
# ============================================================

columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

original_suffixes = (
    "_mean",
    "_std",
    "_min",
    "_max",
    "_range",
    "_rms",
)

original_features = [
    col for col in columns
    if col.endswith(original_suffixes)
]

delta_features = [
    col for col in columns
    if col.endswith("_delta")
    and not col.endswith("_abs_delta")
]

absolute_delta_features = [
    col for col in columns
    if col.endswith("_abs_delta")
]

change_rate_features = [
    col for col in columns
    if col.endswith("_change_rate")
]

feature_columns = list(dict.fromkeys(
    original_features
    + delta_features
    + absolute_delta_features
    + change_rate_features
))

expected_groups = [
    ("Original IMU", original_features),
    ("Delta", delta_features),
    ("Absolute delta", absolute_delta_features),
    ("Change rate", change_rate_features),
]

for name, features in expected_groups:
    if len(features) != 72:
        raise ValueError(
            f"Expected 72 {name} features; found {len(features)}."
        )

if len(feature_columns) != 288:
    raise ValueError(
        f"Expected 288 unique features; found {len(feature_columns)}."
    )

forbidden = {
    "slip",
    "max_contact_speed",
    "slip_fraction",
    "time_start",
    "time_end",
    "window_duration",
    "object",
    "trial",
}

if forbidden.intersection(feature_columns):
    raise ValueError("Leakage or metadata detected in model features.")

print(f"\nTotal model features: {len(feature_columns)}")

# ============================================================
# 4. LOAD FEATURE RANKINGS
# ============================================================

importance_df = pd.read_csv(IMPORTANCE_PATH)

required_importance_columns = {"feature", "importance"}

if not required_importance_columns.issubset(importance_df.columns):
    raise ValueError(
        "Feature-importance CSV must contain feature and importance columns."
    )

if importance_df["feature"].duplicated().any():
    raise ValueError("Duplicate features in feature-importance file.")

if set(importance_df["feature"]) != set(feature_columns):
    raise ValueError(
        "The importance file does not match the 288 current model features."
    )

print("Existing feature-importance file validated.")
print(
    "Note: its ranking is used for reporting only; "
    "cross-validation rankings are recalculated inside each fold."
)

# ============================================================
# 5. LOAD THE DATASET
# ============================================================

print("\nLoading temporal dataset...")

use_columns = feature_columns + ["trial", "slip"]

dtype_map = {
    col: np.float32
    for col in feature_columns
}

dtype_map["trial"] = str
dtype_map["slip"] = np.int8

df = pd.read_csv(
    DATA_PATH,
    usecols=use_columns,
    dtype=dtype_map,
)

if df[feature_columns].isna().any().any():
    raise ValueError("Missing values found in input features.")

if not np.isfinite(
    df[feature_columns].to_numpy(dtype=np.float32)
).all():
    raise ValueError("NaN or infinite feature values detected.")

if df["slip"].isna().any():
    raise ValueError("Missing target labels detected.")

dataset_trials = set(df["trial"].unique())

unassigned = dataset_trials - train_trials - test_trials

if unassigned:
    raise ValueError(
        f"Trials absent from saved split: {list(unassigned)[:5]}"
    )

train_mask = df["trial"].isin(train_trials).to_numpy()
test_mask = df["trial"].isin(test_trials).to_numpy()

X_all = df[feature_columns].to_numpy(
    dtype=np.float32,
    copy=True,
)

y_all = df["slip"].to_numpy(
    dtype=np.int8,
    copy=True,
)

groups_all = df["trial"].to_numpy(copy=True)

X_train = X_all[train_mask]
y_train = y_all[train_mask]
groups_train = groups_all[train_mask]

X_test = X_all[test_mask]
y_test = y_all[test_mask]

del X_all, y_all, groups_all, df
gc.collect()

print(f"Training windows: {len(y_train):,}")
print(f"Testing windows:  {len(y_test):,}")
print(f"Training feature matrix: {X_train.shape}")
print(f"Testing feature matrix:  {X_test.shape}")

# ============================================================
# 6. GROUPED CROSS-VALIDATION
# ============================================================

print("\nStarting five-fold trial-grouped cross-validation.")
print("Feature rankings are learned independently in each fold.")

cv = GroupKFold(n_splits=N_SPLITS)

cv_records = []

start_cv = time.time()

for fold_number, (fit_idx, val_idx) in enumerate(
    cv.split(X_train, y_train, groups_train),
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

    if set(groups_fit).intersection(set(groups_val)):
        raise ValueError(
            f"Trial leakage detected in fold {fold_number}."
        )

    # --------------------------------------------------------
    # 6A. Train a ranking model using only the fold's fit data.
    # --------------------------------------------------------

    ranking_model = CatBoostClassifier(**CATBOOST_PARAMS)

    ranking_model.fit(X_fit, y_fit)

    fold_importance = ranking_model.get_feature_importance()

    ranking_indices = np.argsort(
        fold_importance
    )[::-1]

    del ranking_model
    gc.collect()

    # --------------------------------------------------------
    # 6B. Evaluate each feature count on the fold validation set.
    # --------------------------------------------------------

    for feature_count in FEATURE_COUNTS:
        selected_indices = ranking_indices[:feature_count]

        X_fit_selected = X_fit[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]

        model = CatBoostClassifier(**CATBOOST_PARAMS)

        model.fit(
            X_fit_selected,
            y_fit,
        )

        val_probabilities = model.predict_proba(
            X_val_selected
        )[:, 1]

        val_predictions = (
            val_probabilities >= THRESHOLD
        ).astype(np.int8)

        record = {
            "fold": fold_number,
            "feature_count": feature_count,
            "accuracy": accuracy_score(
                y_val, val_predictions
            ),
            "precision": precision_score(
                y_val,
                val_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_val,
                val_predictions,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_val,
                val_predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_val,
                val_probabilities,
            ),
            "pr_auc": average_precision_score(
                y_val,
                val_probabilities,
            ),
        }

        cv_records.append(record)

        print(
            f"Features={feature_count:3d} | "
            f"Accuracy={record['accuracy']:.4f} | "
            f"F1={record['f1_score']:.4f} | "
            f"ROC-AUC={record['roc_auc']:.4f}"
        )

        del model
        del val_probabilities
        del val_predictions
        del X_fit_selected
        del X_val_selected
        gc.collect()

    del X_fit, X_val, y_fit, y_val
    gc.collect()

cv_seconds = time.time() - start_cv

print(
    f"\nCross-validation completed in "
    f"{cv_seconds / 60:.2f} minutes."
)

# ============================================================
# 7. SUMMARIZE CROSS-VALIDATION RESULTS
# ============================================================

cv_df = pd.DataFrame(cv_records)

cv_df.to_csv(
    RESULT_DIR / "cross_validation_fold_results.csv",
    index=False,
)

summary = (
    cv_df.groupby("feature_count")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1_score", "mean"),
        f1_std=("f1_score", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    RESULT_DIR / "feature_count_comparison.csv",
    index=False,
)

print("\n" + "=" * 75)
print("CROSS-VALIDATION SUMMARY")
print("=" * 75)

print(
    summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}",
    )
)

# Select the feature count with the highest mean CV F1.
# In a tie, prefer the smaller feature set.
summary = summary.sort_values(
    ["f1_mean", "feature_count"],
    ascending=[False, True],
)

best_feature_count = int(
    summary.iloc[0]["feature_count"]
)

print(
    f"\nSelected feature count based on CV F1: "
    f"{best_feature_count}"
)

# ============================================================
# 8. TRAIN FINAL RANKING MODEL ON ALL TRAINING TRIALS
# ============================================================

print("\nTraining final feature-ranking model...")

final_ranking_model = CatBoostClassifier(
    **CATBOOST_PARAMS
)

final_ranking_model.fit(
    X_train,
    y_train,
)

final_importance = final_ranking_model.get_feature_importance()

ranking_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": final_importance,
})

ranking_df = ranking_df.sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

ranking_df.to_csv(
    RESULT_DIR / "training_feature_importance.csv",
    index=False,
)

selected_features = ranking_df["feature"].head(
    best_feature_count
).tolist()

selected_feature_indices = [
    feature_columns.index(feature)
    for feature in selected_features
]

with open(
    RESULT_DIR / "selected_features.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "selected_feature_count": best_feature_count,
            "selection_metric": "mean_grouped_cv_f1",
            "selected_features": selected_features,
        },
        file,
        indent=4,
    )

del final_ranking_model
gc.collect()

# ============================================================
# 9. TRAIN THE FINAL SELECTED MODEL
# ============================================================

print("\nTraining final model with selected features...")

X_train_selected = X_train[:, selected_feature_indices]
X_test_selected = X_test[:, selected_feature_indices]

final_model = CatBoostClassifier(
    **CATBOOST_PARAMS
)

start_training = time.time()

final_model.fit(
    X_train_selected,
    y_train,
)

final_training_seconds = time.time() - start_training

model_path = (
    MODEL_DIR
    / f"catboost_selected_{best_feature_count}_features.cbm"
)

final_model.save_model(str(model_path))

print(f"Final model saved: {model_path}")

# ============================================================
# 10. EVALUATE ON THE HELD-OUT TEST SET
# ============================================================

print("\nEvaluating the selected model on held-out test trials...")

test_probabilities = final_model.predict_proba(
    X_test_selected
)[:, 1]

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(np.int8)

test_metrics = {
    "model": "CatBoost Feature Selected",
    "feature_count": best_feature_count,
    "training_windows": int(len(y_train)),
    "testing_windows": int(len(y_test)),
    "training_trials": int(len(train_trials)),
    "testing_trials": int(len(test_trials)),
    "threshold": THRESHOLD,
    "cross_validation_seconds": round(cv_seconds, 3),
    "final_training_seconds": round(
        final_training_seconds, 3
    ),
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "f1_score": float(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_test,
            test_probabilities,
        )
    ),
    "pr_auc": float(
        average_precision_score(
            y_test,
            test_probabilities,
        )
    ),
}

with open(
    RESULT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        test_metrics,
        file,
        indent=4,
    )

print("\n" + "=" * 75)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 75)

for name, value in test_metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.6f}")
    else:
        print(f"{name}: {value}")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

print("\nConfusion matrix:")
print(cm)

report = classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["No Slip", "Slip"],
    zero_division=0,
)

print("\nClassification report:")
print(report)

with open(
    RESULT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

pd.DataFrame(
    cm,
    index=["Actual No Slip", "Actual Slip"],
    columns=["Predicted No Slip", "Predicted Slip"],
).to_csv(
    RESULT_DIR / "confusion_matrix.csv"
)

# Save test predictions.
pd.DataFrame({
    "actual_slip": y_test,
    "predicted_slip": test_predictions,
    "slip_probability": test_probabilities,
}).to_csv(
    RESULT_DIR / "test_predictions.csv",
    index=False,
)

# ============================================================
# 11. PLOT CROSS-VALIDATION RESULTS
# ============================================================

plot_df = summary.sort_values("feature_count")

fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(
    plot_df["feature_count"],
    plot_df["f1_mean"],
    yerr=plot_df["f1_std"],
    marker="o",
    capsize=4,
)

ax.set_xlabel("Number of Selected Features")
ax.set_ylabel("Mean Cross-Validation F1-score")
ax.set_title("CatBoost Feature Selection: Grouped CV")
ax.set_xticks(FEATURE_COUNTS)
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig(
    RESULT_DIR / "feature_count_vs_f1.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

# Plot final model feature importance.
final_model_importance = final_model.get_feature_importance()

final_importance_df = pd.DataFrame({
    "feature": selected_features,
    "importance": final_model_importance,
}).sort_values(
    "importance",
    ascending=False,
)

final_importance_df.to_csv(
    RESULT_DIR / "selected_model_feature_importance.csv",
    index=False,
)

print("\n" + "=" * 75)
print("FEATURE SELECTION COMPLETED")
print("=" * 75)

print(f"Selected feature count: {best_feature_count}")
print(f"Final model: {model_path}")
print(f"Results directory: {RESULT_DIR}")

print("\nTest metrics:")
print(f"Accuracy:  {test_metrics['accuracy']:.4%}")
print(f"Precision: {test_metrics['precision']:.4%}")
print(f"Recall:    {test_metrics['recall']:.4%}")
print(f"F1-score:  {test_metrics['f1_score']:.4%}")
print(f"ROC-AUC:   {test_metrics['roc_auc']:.4%}")
print(f"PR-AUC:    {test_metrics['pr_auc']:.4%}")