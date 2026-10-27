"""
LightGBM model training for the SLID-CH slip detection dataset.

Features:
- Uses the shared trial-level train/test split.
- Uses only the 72 IMU statistical features.
- Excludes label-derived features to prevent data leakage.
- Trains a LightGBM binary classifier.
- Evaluates accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
- Saves the model, metrics, classification report, confusion matrix,
  test predictions, and feature importance.

Run from the project root:
    python src/train_lightgbm.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

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

MODEL_DIR = PROJECT_ROOT / "models" / "lightgbm"
RESULT_DIR = PROJECT_ROOT / "results" / "lightgbm"

MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

LIGHTGBM_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():
    """Load the dataset and select the 72 IMU statistical features."""

    print("\n" + "=" * 65)
    print("SLID-CH: LIGHTGBM SLIP DETECTION")
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
        raise ValueError("Missing values found in the IMU features.")

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
    print(
        y.value_counts()
        .sort_index()
        .rename({
            0: "No Slip (0)",
            1: "Slip (1)",
        })
    )

    return df, X, y, groups, feature_columns


# ============================================================
# 4. LOAD SHARED TRIAL-LEVEL SPLIT
# ============================================================

def load_trial_split(groups, y):
    """
    Load the same train/test trial split used by the other models.

    If the split file does not exist, create and save one.
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
                "Saved split does not match the current dataset."
            )

    else:

        print("\nShared split not found. Creating a trial-level split.")

        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        train_idx, test_idx = next(
            splitter.split(
                np.zeros(len(y)),
                y,
                groups=groups,
            )
        )

        train_trials = set(groups.iloc[train_idx].unique())
        test_trials = set(groups.iloc[test_idx].unique())

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
        raise ValueError("Train/test split contains an empty partition.")

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
# 5. TRAIN LIGHTGBM
# ============================================================

def train_model(X_train, y_train):
    """Train the LightGBM classifier."""

    print("\n" + "=" * 65)
    print("TRAINING LIGHTGBM")
    print("=" * 65)

    print(f"Training samples: {len(X_train):,}")
    print(f"Number of features: {X_train.shape[1]}")

    print("\nLightGBM parameters:")

    for name, value in LIGHTGBM_PARAMETERS.items():
        print(f"  {name}: {value}")

    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMETERS)

    print("\nTraining started...")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print(
        f"\nTraining completed in {training_time:.2f} seconds."
    )

    return model, training_time


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model using standard metrics."""

    print("\n" + "=" * 65)
    print("EVALUATING LIGHTGBM")
    print("=" * 65)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "LightGBM",
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
# 7. SAVE MODEL AND RESULTS
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
    """Save model, evaluation metrics, predictions, and plots."""

    print("\n" + "=" * 65)
    print("SAVING LIGHTGBM OUTPUTS")
    print("=" * 65)

    metrics["training_time_seconds"] = float(training_time)

    # Save the model with its feature-column order.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "model_name": "LightGBM",
            "parameters": model.get_params(),
        },
        MODEL_PATH,
    )

    print(f"Saved model: {MODEL_PATH}")

    # Save metrics.
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

    # Save confusion matrix as CSV.
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No_Slip", "Actual_Slip"],
        columns=["Predicted_No_Slip", "Predicted_Slip"],
    )

    cm_df.to_csv(RESULT_DIR / "confusion_matrix.csv")

    # Plot confusion matrix.
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

    ax.set_title("LightGBM - Confusion Matrix")

    fig.tight_layout()

    cm_image_path = RESULT_DIR / "confusion_matrix.png"

    fig.savefig(
        cm_image_path,
        dpi=200,
    )

    plt.close(fig)

    # Save predictions.
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
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False,
    )

    # Plot top 20 features.
    top_features = importance_df.head(20).sort_values(
        "importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top_features["feature"],
        top_features["importance"],
    )

    ax.set_title("LightGBM - Top 20 Feature Importances")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("IMU Feature")

    fig.tight_layout()

    importance_image_path = RESULT_DIR / "feature_importance.png"

    fig.savefig(
        importance_image_path,
        dpi=200,
    )

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

    train_indices, test_indices = load_trial_split(
        groups,
        y,
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    test_df = df.iloc[test_indices].copy()

    model, training_time = train_model(
        X_train,
        y_train,
    )

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
    print("LIGHTGBM TRAINING FINISHED")
    print("=" * 65)

    print("\nFinal metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(
        f"Train time: "
        f"{metrics['training_time_seconds']:.2f} seconds"
    )


if __name__ == "__main__":
    main()