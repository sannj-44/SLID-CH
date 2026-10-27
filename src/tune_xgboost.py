"""
XGBoost Hyperparameter Tuning for the SLID-CH Slip Detection Dataset.

Features:
    - Uses 72 IMU statistical features.
    - Reuses the saved trial-level train/test split.
    - Uses GroupKFold to prevent windows from the same trial
      appearing in both training and validation folds.
    - Uses RandomizedSearchCV for hyperparameter tuning.
    - Limits parallel processing to reduce memory usage.
    - Saves the tuned model and evaluation results separately.
    - Does not modify the original dataset or baseline model.

Run from the project root:
    python src/tune_xgboost.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
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

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

# Number of hyperparameter combinations to evaluate.
N_ITER = 20

# Run CV fits sequentially to reduce RAM consumption.
SEARCH_N_JOBS = 1

# Limit CPU threads used by each XGBoost fit.
XGBOOST_N_JOBS = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "slid_ch_features.csv"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "trial_split.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "xgboost_tuned"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "xgboost_tuned"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET AND SAVED SPLIT
# ============================================================

def load_dataset():
    """Load the processed dataset and original trial split."""

    print("\n" + "=" * 70)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Trial split file not found:\n{SPLIT_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split_info = json.load(file)

    required_columns = {"trial", "slip"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(f"\nDataset shape: {df.shape}")

    return df, split_info


# ============================================================
# 3. SELECT THE 72 IMU FEATURES
# ============================================================

def select_features(df):
    """
    Select only the 72 statistical IMU features.

    Excluded:
        slip               - target
        max_contact_speed  - label-derived information
        slip_fraction      - label-derived information
        time_start         - metadata
        time_end           - metadata
        window_duration    - metadata
        object             - object metadata
        trial              - grouping metadata
    """

    valid_suffixes = (
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
        if column.endswith(valid_suffixes)
    ]

    if len(feature_columns) != 72:
        raise ValueError(
            f"Expected 72 IMU features, found "
            f"{len(feature_columns)}."
        )

    X_df = df[feature_columns].copy()

    X_df = X_df.apply(
        pd.to_numeric,
        errors="raise",
    )

    if X_df.isna().any().any():
        raise ValueError(
            "Missing values detected in IMU features."
        )

    X = np.ascontiguousarray(
        X_df.to_numpy(dtype=np.float32)
    )

    y = np.ascontiguousarray(
        pd.to_numeric(
            df["slip"],
            errors="raise",
        ).to_numpy(dtype=np.int32)
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values detected in IMU features."
        )

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            "The target must contain only labels 0 and 1."
        )

    print(f"\nNumber of features: {len(feature_columns)}")

    unique_labels, label_counts = np.unique(
        y,
        return_counts=True,
    )

    print("\nTarget distribution:")

    for label, count in zip(unique_labels, label_counts):
        label_name = "No Slip" if label == 0 else "Slip"

        print(
            f"{label_name} ({label}): {count:,} "
            f"({count / len(y) * 100:.2f}%)"
        )

    return X, y, feature_columns


# ============================================================
# 4. CREATE THE ORIGINAL TRIAL-LEVEL TRAIN/TEST SPLIT
# ============================================================

def create_train_test_split(
    df,
    X,
    y,
    split_info,
):
    """
    Reuse the saved split.

    No trial is allowed to appear in both train and test sets.
    """

    if (
        "train_trials" not in split_info
        or "test_trials" not in split_info
    ):
        raise ValueError(
            "trial_split.json must contain train_trials "
            "and test_trials."
        )

    train_trials = set(
        map(str, split_info["train_trials"])
    )

    test_trials = set(
        map(str, split_info["test_trials"])
    )

    if train_trials & test_trials:
        raise ValueError(
            "Train/test trial overlap detected in split file."
        )

    trial_ids = df["trial"].astype(str).to_numpy()

    train_mask = np.isin(
        trial_ids,
        list(train_trials),
    )

    test_mask = np.isin(
        trial_ids,
        list(test_trials),
    )

    if np.any(train_mask & test_mask):
        raise ValueError(
            "A trial appears in both train and test sets."
        )

    if np.any(~(train_mask | test_mask)):
        unmatched = np.unique(
            trial_ids[~(train_mask | test_mask)]
        )

        raise ValueError(
            "Some dataset trials are missing from the saved split. "
            f"Examples: {unmatched[:10].tolist()}"
        )

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    train_groups = trial_ids[train_mask]
    test_groups = trial_ids[test_mask]

    if set(train_groups) & set(test_groups):
        raise ValueError(
            "Train/test trial leakage detected."
        )

    print("\nTrain/test split")
    print("-" * 50)

    print(f"Training windows: {len(X_train):,}")
    print(f"Testing windows:  {len(X_test):,}")

    print(
        f"Training trials:  "
        f"{len(np.unique(train_groups))}"
    )

    print(
        f"Testing trials:   "
        f"{len(np.unique(test_groups))}"
    )

    print("\nTraining classes:")

    for label in [0, 1]:
        count = int(np.sum(y_train == label))

        print(
            f"{'No Slip' if label == 0 else 'Slip'}: "
            f"{count:,}"
        )

    print("\nTesting classes:")

    for label in [0, 1]:
        count = int(np.sum(y_test == label))

        print(
            f"{'No Slip' if label == 0 else 'Slip'}: "
            f"{count:,}"
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_groups,
        test_groups,
    )


# ============================================================
# 5. CREATE THE XGBOOST SEARCH
# ============================================================

def create_search(scale_pos_weight):
    """Create the classifier and randomized parameter search."""

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=XGBOOST_N_JOBS,
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
    )

    parameter_distributions = {
        "n_estimators": [
            200,
            300,
            400,
            500,
        ],

        "max_depth": [
            3,
            4,
            5,
            6,
            7,
            8,
        ],

        "learning_rate": [
            0.03,
            0.05,
            0.07,
            0.1,
        ],

        "min_child_weight": [
            1,
            3,
            5,
            7,
            10,
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "colsample_bytree": [
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "gamma": [
            0,
            0.1,
            0.2,
            0.5,
            1.0,
        ],

        "reg_alpha": [
            0,
            0.01,
            0.1,
            0.5,
            1.0,
        ],

        "reg_lambda": [
            1,
            2,
            5,
            10,
        ],
    }

    # Explicit GroupKFold is essential.
    # Do not leave cv=None because that would use the default splitter.
    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_distributions,
        n_iter=N_ITER,
        scoring="accuracy",
        cv=cv,
        n_jobs=SEARCH_N_JOBS,
        random_state=RANDOM_STATE,
        verbose=2,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    return search


# ============================================================
# 6. RUN HYPERPARAMETER TUNING
# ============================================================

def tune_model(
    X_train,
    y_train,
    train_groups,
):
    """Tune XGBoost using trial-grouped cross-validation."""

    print("\n" + "=" * 70)
    print("STARTING RANDOMIZED HYPERPARAMETER SEARCH")
    print("=" * 70)

    print(f"Candidates: {N_ITER}")
    print(f"CV folds:   {N_SPLITS}")
    print(f"Total fits: {N_ITER * N_SPLITS}")

    # Verify every fold has both target classes.
    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    for fold_number, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(
            X_train,
            y_train,
            groups=train_groups,
        ),
        start=1,
    ):
        train_classes = set(
            np.unique(y_train[train_idx]).tolist()
        )

        val_classes = set(
            np.unique(y_train[val_idx]).tolist()
        )

        print(
            f"Fold {fold_number}: "
            f"train={len(train_idx):,}, "
            f"validation={len(val_idx):,}, "
            f"train classes={sorted(train_classes)}, "
            f"validation classes={sorted(val_classes)}"
        )

        if train_classes != {0, 1}:
            raise ValueError(
                f"Training fold {fold_number} "
                "does not contain both classes."
            )

        if val_classes != {0, 1}:
            raise ValueError(
                f"Validation fold {fold_number} "
                "does not contain both classes."
            )

        # Verify no trial is shared by training and validation.
        fold_train_groups = set(
            train_groups[train_idx]
        )

        fold_val_groups = set(
            train_groups[val_idx]
        )

        if fold_train_groups & fold_val_groups:
            raise ValueError(
                f"Trial leakage detected in fold {fold_number}."
            )

    class_counts = np.bincount(
        y_train,
        minlength=2,
    )

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError(
            "Training data must contain both classes."
        )

    scale_pos_weight = (
        class_counts[0] / class_counts[1]
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.6f}"
    )

    search = create_search(
        scale_pos_weight
    )

    print("\nStarting search.")
    print("Cross-validation fits will run sequentially.")
    print("Monitor RAM usage in Windows Task Manager.")

    start_time = time.time()

    search.fit(
        X_train,
        y_train,
        groups=train_groups,
    )

    elapsed_time = (
        time.time() - start_time
    )

    print("\nSearch completed.")

    print(
        f"Search duration: "
        f"{elapsed_time / 60:.2f} minutes"
    )

    print(
        f"Best CV accuracy: "
        f"{search.best_score_:.6f}"
    )

    print("\nBest parameters:")

    for parameter, value in search.best_params_.items():
        print(f"  {parameter}: {value}")

    # Save all cross-validation results.
    cv_results = pd.DataFrame(
        search.cv_results_
    )

    cv_results = cv_results.sort_values(
        by="rank_test_score"
    ).reset_index(drop=True)

    cv_results.to_csv(
        RESULT_DIR / "cross_validation_results.csv",
        index=False,
    )

    # Save best parameters and search metadata.
    best_parameters = {
        "best_parameters": search.best_params_,
        "best_cv_accuracy": float(
            search.best_score_
        ),
        "search_duration_seconds": float(
            elapsed_time
        ),
        "n_iter": N_ITER,
        "n_splits": N_SPLITS,
        "scale_pos_weight": float(
            scale_pos_weight
        ),
    }

    with open(
        RESULT_DIR / "best_parameters.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_parameters,
            file,
            indent=4,
        )

    return search, elapsed_time


# ============================================================
# 7. EVALUATE THE SELECTED MODEL
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_test,
    train_groups,
    test_groups,
):
    """
    Evaluate the selected model on the held-out test set.

    The test set is not used for hyperparameter selection.
    """

    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 70)

    start_time = time.time()

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction_time = (
        time.time() - start_time
    )

    # Standard classification threshold.
    threshold = 0.5

    y_pred = (
        y_probability >= threshold
    ).astype(np.int32)

    metrics = {
        "model": "XGBoost Tuned",

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

        "decision_threshold": threshold,

        "test_prediction_time_seconds": float(
            prediction_time
        ),

        "train_windows": int(
            len(X_train)
        ),

        "test_windows": int(
            len(X_test)
        ),

        "train_trials": int(
            len(np.unique(train_groups))
        ),

        "test_trials": int(
            len(np.unique(test_groups))
        ),
    }

    print("\nMetrics:")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{name:35s}: {value:.6f}"
            )
        else:
            print(
                f"{name:35s}: {value}"
            )

    # Classification report.
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No Slip",
            "Slip",
        ],
        digits=4,
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

    # Save evaluation metrics.
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

    # Confusion matrix.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    print("\nConfusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Slip",
            "Slip",
        ],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "Tuned XGBoost - Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return metrics


# ============================================================
# 8. SAVE THE MODEL AND FEATURE IMPORTANCE
# ============================================================

def save_model_and_importance(
    model,
    feature_columns,
):
    """Save the tuned model and feature importance."""

    model_path = (
        MODEL_DIR
        / "xgboost_tuned_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nSaved tuned model:\n{model_path}"
    )

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False,
    )

    # Plot the top 20 features.
    top_n = min(
        20,
        len(importance_df),
    )

    plot_df = importance_df.head(
        top_n
    ).sort_values(
        by="importance",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["importance"],
    )

    ax.set_title(
        "Tuned XGBoost - Top 20 Feature Importances"
    )

    ax.set_xlabel(
        "Feature Importance"
    )

    ax.set_ylabel(
        "IMU Feature"
    )

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved feature importance CSV:\n"
        f"{RESULT_DIR / 'feature_importance.csv'}"
    )

    print(
        f"Saved feature importance plot:\n"
        f"{RESULT_DIR / 'feature_importance.png'}"
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():
    total_start = time.time()

    # Load dataset and saved trial split.
    df, split_info = load_dataset()

    # Select features and target.
    X, y, feature_columns = select_features(
        df
    )

    # Reuse the original train/test split.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_groups,
        test_groups,
    ) = create_train_test_split(
        df,
        X,
        y,
        split_info,
    )

    # Tune the model.
    search, search_time = tune_model(
        X_train,
        y_train,
        train_groups,
    )

    best_model = search.best_estimator_

    # Save the tuned model separately.
    save_model_and_importance(
        best_model,
        feature_columns,
    )

    # Evaluate the selected model.
    metrics = evaluate_model(
        best_model,
        X_train,
        X_test,
        y_test,
        train_groups,
        test_groups,
    )

    # Save final summary.
    summary = {
        "model": "XGBoost Tuned",

        "best_cv_accuracy": float(
            search.best_score_
        ),

        "test_accuracy": metrics["accuracy"],

        "test_precision": metrics["precision"],

        "test_recall": metrics["recall"],

        "test_f1_score": metrics["f1_score"],

        "test_roc_auc": metrics["roc_auc"],

        "test_pr_auc": metrics["pr_auc"],

        "search_duration_seconds": float(
            search_time
        ),

        "total_duration_seconds": float(
            time.time() - total_start
        ),

        "train_trials": int(
            len(np.unique(train_groups))
        ),

        "test_trials": int(
            len(np.unique(test_groups))
        ),

        "n_features": len(
            feature_columns
        ),

        "n_search_candidates": N_ITER,

        "cv_folds": N_SPLITS,
    }

    with open(
        RESULT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    # Final summary.
    print("\n" + "=" * 70)
    print("XGBOOST HYPERPARAMETER TUNING COMPLETED")
    print("=" * 70)

    print(
        f"Best CV accuracy: "
        f"{search.best_score_:.4%}"
    )

    print(
        f"Test accuracy:    "
        f"{metrics['accuracy']:.4%}"
    )

    print(
        f"Test precision:   "
        f"{metrics['precision']:.4%}"
    )

    print(
        f"Test recall:      "
        f"{metrics['recall']:.4%}"
    )

    print(
        f"Test F1-score:    "
        f"{metrics['f1_score']:.4%}"
    )

    print(
        f"Test ROC-AUC:     "
        f"{metrics['roc_auc']:.4%}"
    )

    print(
        f"Test PR-AUC:      "
        f"{metrics['pr_auc']:.4%}"
    )

    print(
        f"\nModel directory:\n{MODEL_DIR}"
    )

    print(
        f"\nResults directory:\n{RESULT_DIR}"
    )

    print("\nImportant:")
    print(
        "The saved test set is for final evaluation, not "
        "hyperparameter selection. Repeatedly choosing models "
        "based on test scores can make those scores optimistic."
    )

    print(
        "The current slip labels are derived from contact sliding "
        "velocity thresholds. Model performance measures agreement "
        "with those labels, which should be validated against the "
        "dataset's official annotation definition."
    )


if __name__ == "__main__":
    main()