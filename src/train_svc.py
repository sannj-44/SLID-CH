"""
SVC model training for the SLID-CH slip detection dataset.

Features:
- Uses the shared trial-level train/test split.
- Uses only the 72 IMU statistical features.
- Excludes label-derived features to prevent data leakage.
- Uses a stratified subset of training windows for practical runtime.
- Evaluates on the complete, unchanged test set.
- Saves the trained pipeline, metrics, classification report,
  confusion matrix, test predictions, and evaluation plots.

Run from the project root:
    python src/train_svc.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "slid_ch_features.csv"
SPLIT_PATH = PROJECT_ROOT / "results" / "trial_split.json"

MODEL_DIR = PROJECT_ROOT / "models" / "svc"
RESULT_DIR = PROJECT_ROOT / "results" / "svc"

MODEL_PATH = MODEL_DIR / "svc_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Start with 50,000 training windows.
# Increase this only if training time and RAM usage are acceptable.
MAX_TRAIN_SAMPLES = 50_000

SVC_PARAMETERS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": RANDOM_STATE,
    "cache_size": 2000,
}


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():
    """Load the dataset and select the 72 IMU statistical features."""

    print("\n" + "=" * 65)
    print("SLID-CH: SVC SLIP DETECTION")
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
            f"Expected 72 IMU features, found {len(feature_columns)}."
        )

    X = df[feature_columns].astype(np.float32)
    y = df["slip"].astype(np.int8)
    groups = df["trial"].astype(str)

    if X.isnull().any().any():
        raise ValueError("Missing values found in IMU features.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite or non-finite IMU feature values found."
        )

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "The target must contain only binary labels 0 and 1."
        )

    print(f"Number of IMU features: {len(feature_columns)}")
    print(f"Number of windows: {len(df):,}")
    print(f"Number of trials: {groups.nunique()}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    return df, X, y, groups, feature_columns


# ============================================================
# 4. LOAD SHARED TRIAL-LEVEL SPLIT
# ============================================================

def load_trial_split(groups, y):
    """Load the shared split used by the other models."""

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Shared split not found: {SPLIT_PATH}\n"
            "Run one of the existing training scripts first."
        )

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
            "Saved split does not match the current dataset."
        )

    train_indices = np.flatnonzero(
        groups.isin(train_trials).to_numpy()
    )

    test_indices = np.flatnonzero(
        groups.isin(test_trials).to_numpy()
    )

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Train/test split contains an empty partition.")

    train_trial_ids = set(groups.iloc[train_indices])
    test_trial_ids = set(groups.iloc[test_indices])

    if train_trial_ids & test_trial_ids:
        raise ValueError(
            "Trial leakage detected between train and test."
        )

    print("\nTrial-level split:")
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials:  {len(test_trials)}")
    print(f"Training windows before sampling: {len(train_indices):,}")
    print(f"Testing windows: {len(test_indices):,}")
    print("Shared trials between train and test: 0")

    print("\nFull training label distribution:")
    print(y.iloc[train_indices].value_counts().sort_index())

    print("\nTesting label distribution:")
    print(y.iloc[test_indices].value_counts().sort_index())

    return train_indices, test_indices


# ============================================================
# 5. SELECT STRATIFIED TRAINING SUBSET
# ============================================================

def select_training_subset(X_train, y_train):
    """
    Select a stratified subset of training windows.

    The complete test set remains unchanged.

    Stratification approximately preserves the overall class
    proportions in the training subset.
    """

    if len(X_train) <= MAX_TRAIN_SAMPLES:
        print("\nTraining set is within the configured sample limit.")
        return X_train, y_train

    print("\n" + "=" * 65)
    print("SELECTING SVC TRAINING SUBSET")
    print("=" * 65)

    print(f"Original training windows: {len(X_train):,}")
    print(f"Configured maximum: {MAX_TRAIN_SAMPLES:,}")

    X_subset, _, y_subset, _ = train_test_split(
        X_train,
        y_train,
        train_size=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"Selected training windows: {len(X_subset):,}")

    print("\nSubset class distribution:")
    print(y_subset.value_counts().sort_index())

    print(
        "\nNote: the subset is stratified by label, "
        "not sampled by trial."
    )

    return X_subset, y_subset


# ============================================================
# 6. TRAIN SVC
# ============================================================

def train_model(X_train, y_train):
    """Train the scaled RBF-kernel SVC pipeline."""

    print("\n" + "=" * 65)
    print("TRAINING SVC")
    print("=" * 65)

    print(f"Training samples: {len(X_train):,}")
    print(f"Number of features: {X_train.shape[1]}")

    print("\nSVC parameters:")

    for name, value in SVC_PARAMETERS.items():
        print(f"  {name}: {value}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**SVC_PARAMETERS)),
    ])

    print("\nTraining started...")
    print(
        "RBF SVC may take significantly longer than tree-based models."
    )

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds.")

    return model, training_time


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate the SVC on the complete test set."""

    print("\n" + "=" * 65)
    print("EVALUATING SVC")
    print("=" * 65)

    y_pred = model.predict(X_test).astype(int)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "SVC",
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
            roc_auc_score(y_test, y_prob)
        ),
        "pr_auc": float(
            average_precision_score(y_test, y_prob)
        ),
        "test_samples": int(len(y_test)),
        "test_no_slip_samples": int((y_test == 0).sum()),
        "test_slip_samples": int((y_test == 1).sum()),
        "training_subset_size": int(
            model.named_steps["svc"].shape_fit_[0]
        ),
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

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    print("Confusion matrix:")
    print(cm)

    return metrics, y_pred, y_prob, report, cm


# ============================================================
# 8. SAVE MODEL AND RESULTS
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
    """Save the fitted model and all evaluation outputs."""

    print("\n" + "=" * 65)
    print("SAVING SVC OUTPUTS")
    print("=" * 65)

    metrics["training_time_seconds"] = float(training_time)

    # Save the fitted pipeline and feature order.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "model_name": "SVC",
            "parameters": SVC_PARAMETERS,
            "training_subset_size": metrics["training_subset_size"],
        },
        MODEL_PATH,
    )

    print(f"Saved model: {MODEL_PATH}")

    # Save metrics in JSON and CSV.
    metrics_json = RESULT_DIR / "metrics.json"
    metrics_csv = RESULT_DIR / "metrics.csv"

    with open(metrics_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame([metrics]).to_csv(
        metrics_csv,
        index=False,
    )

    # Save classification report.
    report_path = RESULT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # Save confusion matrix CSV.
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No_Slip", "Actual_Slip"],
        columns=["Predicted_No_Slip", "Predicted_Slip"],
    )

    cm_df.to_csv(
        RESULT_DIR / "confusion_matrix.csv"
    )

    # Save confusion matrix plot.
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Slip", "Slip"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("SVC - Confusion Matrix")

    fig.tight_layout()

    cm_image_path = RESULT_DIR / "confusion_matrix.png"

    fig.savefig(
        cm_image_path,
        dpi=200,
    )

    plt.close(fig)

    # Save predictions, including dataset row indices and trial IDs.
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

    print("\nSaved files:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Metrics JSON: {metrics_json}")
    print(f"  Metrics CSV: {metrics_csv}")
    print(f"  Classification report: {report_path}")
    print(f"  Confusion matrix CSV: {RESULT_DIR / 'confusion_matrix.csv'}")
    print(f"  Confusion matrix plot: {cm_image_path}")
    print(f"  Test predictions: {RESULT_DIR / 'test_predictions.csv'}")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    df, X, y, groups, feature_columns = load_dataset()

    train_indices, test_indices = load_trial_split(
        groups,
        y,
    )

    # Split by trial first.
    X_train_full = X.iloc[train_indices]
    y_train_full = y.iloc[train_indices]

    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    test_df = df.iloc[test_indices].copy()

    # Sample only the training data.
    X_train, y_train = select_training_subset(
        X_train_full,
        y_train_full,
    )

    # Train.
    model, training_time = train_model(
        X_train,
        y_train,
    )

    # Evaluate on the complete shared test set.
    metrics, y_pred, y_prob, report, cm = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # Save all outputs.
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
    print("SVC TRAINING FINISHED")
    print("=" * 65)

    print("\nFinal metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(
        f"Training time: "
        f"{metrics['training_time_seconds']:.2f} seconds"
    )


if __name__ == "__main__":
    main()