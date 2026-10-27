"""
XGBoost model training for the SLID-CH slip detection dataset.

Features:
- Uses the same trial-level train/test split as the other models.
- Uses only the 72 IMU statistical features.
- Excludes label-derived features to prevent data leakage.
- Trains an XGBoost binary classifier.
- Evaluates accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
- Saves the trained model, metrics, classification report,
  confusion matrix, test predictions, and feature importance.

Run from the project root:
    python src/train_xgboost.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

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

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "slid_ch_features.csv"
SPLIT_PATH = PROJECT_ROOT / "results" / "trial_split.json"

MODEL_DIR = PROJECT_ROOT / "models" / "xgboost"
RESULT_DIR = PROJECT_ROOT / "results" / "xgboost"

MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

XGB_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": 1,
}


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():
    """Load the prepared dataset and select the 72 IMU features."""

    print("\n" + "=" * 65)
    print("SLID-CH: XGBOOST SLIP DETECTION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Run src/prepare_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset loaded: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")

    required_columns = {"trial", "slip"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    # Select only IMU statistical features.
    # Explicitly exclude label-derived and metadata columns.
    feature_suffixes = (
        "_mean",
        "_std",
        "_min",
        "_max",
        "_range",
        "_rms",
    )

    excluded_columns = {
        "time_start",
        "time_end",
        "window_duration",
        "max_contact_speed",
        "slip_fraction",
        "object",
        "trial",
        "slip",
    }

    feature_columns = [
        column
        for column in df.columns
        if column.endswith(feature_suffixes)
        and column not in excluded_columns
    ]

    if len(feature_columns) != 72:
        raise ValueError(
            f"Expected 72 IMU features, found {len(feature_columns)}.\n"
            f"Selected features: {feature_columns}"
        )

    X = df[feature_columns].astype(np.float32)
    y = df["slip"].astype(np.int8)
    groups = df["trial"].astype(str)

    if X.isnull().any().any():
        raise ValueError("Missing values found in the IMU features.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Infinite or non-finite IMU feature values found.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("The target must contain only binary labels 0 and 1.")

    print(f"Number of IMU features: {len(feature_columns)}")
    print(f"Number of windows: {len(df):,}")
    print(f"Number of trials: {groups.nunique()}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index().rename({
        0: "No Slip (0)",
        1: "Slip (1)",
    }))

    return df, X, y, groups, feature_columns


# ============================================================
# 4. LOAD OR CREATE THE SHARED TRIAL-LEVEL SPLIT
# ============================================================

def load_trial_split(groups, y):
    """
    Load the existing split used by the other models.

    If no split file exists, create one and save it.
    The split is performed by trial, not by individual window.
    """

    if SPLIT_PATH.exists():
        print(f"\nLoading shared split: {SPLIT_PATH}")

        with open(SPLIT_PATH, "r", encoding="utf-8") as file:
            split_info = json.load(file)

        train_trials = set(
            str(trial) for trial in split_info["train_trials"]
        )

        test_trials = set(
            str(trial) for trial in split_info["test_trials"]
        )

        available_trials = set(groups.unique())

        if train_trials & test_trials:
            raise ValueError(
                "Invalid split: a trial appears in both train and test."
            )

        if train_trials | test_trials != available_trials:
            raise ValueError(
                "The saved split does not match the current dataset's "
                "trial IDs. Check results/trial_split.json."
            )

    else:
        print("\nNo shared split file found. Creating a trial-level split.")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        train_indices, test_indices = next(
            splitter.split(
                np.zeros(len(y)),
                y,
                groups=groups,
            )
        )

        train_trials = set(groups.iloc[train_indices].unique())
        test_trials = set(groups.iloc[test_indices].unique())

        split_info = {
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
            "train_trials": sorted(train_trials),
            "test_trials": sorted(test_trials),
        }

        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(SPLIT_PATH, "w", encoding="utf-8") as file:
            json.dump(split_info, file, indent=4)

        print(f"Saved shared split: {SPLIT_PATH}")

    train_mask = groups.isin(train_trials).to_numpy()
    test_mask = groups.isin(test_trials).to_numpy()

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("The train/test split produced an empty partition.")

    if set(groups.iloc[train_indices]) & set(groups.iloc[test_indices]):
        raise ValueError("Trial leakage detected between train and test.")

    print("\nTrial-level split:")
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials:  {len(test_trials)}")
    print(f"Training windows: {len(train_indices):,}")
    print(f"Testing windows:  {len(test_indices):,}")
    print("Shared trials between train and test: 0")

    print("\nTraining label distribution:")
    print(y.iloc[train_indices].value_counts().sort_index())

    print("\nTesting label distribution:")
    print(y.iloc[test_indices].value_counts().sort_index())

    return train_indices, test_indices


# ============================================================
# 5. TRAIN XGBOOST
# ============================================================

def train_model(X_train, y_train):
    """Train the XGBoost classifier."""

    print("\n" + "=" * 65)
    print("TRAINING XGBOOST")
    print("=" * 65)

    # Approximate class balancing using the training-set class counts.
    # This does not use the test labels.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both classes must be present in the training set.")

    scale_pos_weight = negative_count / positive_count

    parameters = XGB_PARAMETERS.copy()
    parameters["scale_pos_weight"] = scale_pos_weight

    print(f"Training samples: {len(X_train):,}")
    print(f"Negative samples: {negative_count:,}")
    print(f"Positive samples: {positive_count:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    print("\nXGBoost parameters:")
    for name, value in parameters.items():
        print(f"  {name}: {value}")

    model = XGBClassifier(**parameters)

    print("\nTraining started...")
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds.")

    return model, training_time


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics and generate a report."""

    print("\n" + "=" * 65)
    print("EVALUATING XGBOOST")
    print("=" * 65)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "XGBoost",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "test_samples": int(len(y_test)),
        "test_no_slip_samples": int((y_test == 0).sum()),
        "test_slip_samples": int((y_test == 1).sum()),
    }

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

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

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("Confusion matrix:")
    print(cm)

    return metrics, y_pred, y_prob, report, cm


# ============================================================
# 7. SAVE RESULTS
# ============================================================

def save_results(
    model,
    feature_columns,
    metrics,
    y_test,
    y_pred,
    y_prob,
    report,
    cm,
    training_time,
    test_df,
):
    """Save the model, metrics, predictions, and visualizations."""

    print("\n" + "=" * 65)
    print("SAVING XGBOOST OUTPUTS")
    print("=" * 65)

    metrics["training_time_seconds"] = float(training_time)

    # Save the model and its feature-column order.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "model_name": "XGBoost",
            "parameters": model.get_params(),
        },
        MODEL_PATH,
    )

    print(f"Saved model: {MODEL_PATH}")

    # Save metrics in JSON and CSV formats.
    metrics_json = RESULT_DIR / "metrics.json"
    metrics_csv = RESULT_DIR / "metrics.csv"

    with open(metrics_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    # Save the classification report.
    report_path = RESULT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # Save confusion matrix as CSV.
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No_Slip", "Actual_Slip"],
        columns=["Predicted_No_Slip", "Predicted_Slip"],
    )

    cm_df.to_csv(RESULT_DIR / "confusion_matrix.csv")

    # Save confusion matrix image.
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Slip", "Slip"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("XGBoost - Confusion Matrix")
    fig.tight_layout()

    cm_image_path = RESULT_DIR / "confusion_matrix.png"
    fig.savefig(cm_image_path, dpi=200)
    plt.close(fig)

    # Save test predictions, preserving the original dataset row indices.
    predictions = pd.DataFrame({
        "dataset_row_index": test_df.index.to_numpy(),
        "trial": test_df["trial"].astype(str).to_numpy(),
        "actual_slip": np.asarray(y_test),
        "predicted_slip": np.asarray(y_pred),
        "slip_probability": np.asarray(y_prob),
    })

    predictions.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False,
    )

    # Save feature importance.
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False,
    )

    # Plot the top 20 features.
    top_features = importance_df.head(20).sort_values(
        "importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top_features["feature"],
        top_features["importance"],
    )

    ax.set_title("XGBoost - Top 20 Feature Importances")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("IMU Feature")

    fig.tight_layout()

    importance_image_path = RESULT_DIR / "feature_importance.png"
    fig.savefig(importance_image_path, dpi=200)
    plt.close(fig)

    print("\nSaved files:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Metrics JSON: {metrics_json}")
    print(f"  Metrics CSV: {metrics_csv}")
    print(f"  Classification report: {report_path}")
    print(f"  Confusion matrix CSV: {RESULT_DIR / 'confusion_matrix.csv'}")
    print(f"  Confusion matrix plot: {cm_image_path}")
    print(f"  Test predictions: {RESULT_DIR / 'test_predictions.csv'}")
    print(f"  Feature importance CSV: {RESULT_DIR / 'feature_importance.csv'}")
    print(f"  Feature importance plot: {importance_image_path}")


# ============================================================
# 8. MAIN
# ============================================================

def main():
    df, X, y, groups, feature_columns = load_dataset()

    train_indices, test_indices = load_trial_split(groups, y)

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    test_df = df.iloc[test_indices].copy()

    model, training_time = train_model(X_train, y_train)

    metrics, y_pred, y_prob, report, cm = evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_results(
        model=model,
        feature_columns=feature_columns,
        metrics=metrics,
        y_test=y_test.to_numpy(),
        y_pred=y_pred,
        y_prob=y_prob,
        report=report,
        cm=cm,
        training_time=training_time,
        test_df=test_df,
    )

    print("\n" + "=" * 65)
    print("XGBOOST TRAINING FINISHED")
    print("=" * 65)

    print("\nFinal metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Train time:{metrics['training_time_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()