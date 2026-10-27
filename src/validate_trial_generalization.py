from pathlib import Path
import json
import warnings

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
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CORRECTED: Use the temporal feature dataset.
FEATURES_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "slid_ch_features_temporal.csv"
)

SPLIT_PATH = PROJECT_ROOT / "results" / "trial_split.json"

SELECTED_FEATURES_PATH = (
    PROJECT_ROOT
    / "results"
    / "catboost_feature_selected"
    / "selected_features.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "catboost_feature_selected"
    / "catboost_selected_150_features.cbm"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "trial_validation"

THRESHOLD = 0.41

TARGET_COLUMN = "slip"
GROUP_COLUMN = "trial"
OBJECT_COLUMN = "object"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_trial_id(value):
    """Normalize trial identifiers for matching across files."""
    if pd.isna(value):
        return None
    return str(value).strip()


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics and confusion-matrix counts."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "windows": int(len(y_true)),
        "no_slip_windows": int(np.sum(y_true == 0)),
        "slip_windows": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob)
        )
        metrics["pr_auc"] = float(
            average_precision_score(y_true, y_prob)
        )
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def save_json(data, path):
    """Save JSON while rejecting invalid numeric values."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, allow_nan=False)


def load_selected_features(path):
    """Load the selected feature list without changing its order."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        features = data

    elif isinstance(data, dict):
        if "selected_features" in data:
            features = data["selected_features"]
        elif "features" in data:
            features = data["features"]
        else:
            raise ValueError(
                "Cannot identify the feature list in selected_features.json. "
                f"Available keys: {list(data.keys())}"
            )
    else:
        raise ValueError(
            "Unexpected format in selected_features.json."
        )

    features = [str(feature) for feature in features]

    if len(features) != 150:
        raise ValueError(
            f"Expected 150 selected features, found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError("Duplicate selected feature names detected.")

    return features


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("INDEPENDENT TRIAL-LEVEL VALIDATION")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    print("\n[1/8] Checking required files...")

    required_files = [
        FEATURES_PATH,
        SPLIT_PATH,
        SELECTED_FEATURES_PATH,
        MODEL_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )
        print(f"[FOUND] {path.relative_to(PROJECT_ROOT)}")

    # --------------------------------------------------------
    # 2. Load trial split
    # --------------------------------------------------------

    print("\n[2/8] Loading saved trial split...")

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split = json.load(file)

    if "train_trials" not in split or "test_trials" not in split:
        raise KeyError(
            "trial_split.json must contain train_trials and test_trials."
        )

    train_trials = {
        normalize_trial_id(value)
        for value in split["train_trials"]
    }

    test_trials = {
        normalize_trial_id(value)
        for value in split["test_trials"]
    }

    train_trials.discard(None)
    test_trials.discard(None)

    overlap = train_trials.intersection(test_trials)

    if overlap:
        raise ValueError(
            f"Trial leakage detected: {len(overlap)} trials occur "
            "in both training and test sets."
        )

    print(f"Training trials: {len(train_trials)}")
    print(f"Test trials:     {len(test_trials)}")
    print(f"Trial overlap:   {len(overlap)}")

    # --------------------------------------------------------
    # 3. Load feature list
    # --------------------------------------------------------

    print("\n[3/8] Loading selected features...")

    selected_features = load_selected_features(
        SELECTED_FEATURES_PATH
    )

    print(f"Selected feature count: {len(selected_features)}")

    # --------------------------------------------------------
    # 4. Load temporal feature dataset
    # --------------------------------------------------------

    print("\n[4/8] Loading temporal feature dataset...")

    print(f"Dataset: {FEATURES_PATH.name}")

    df = pd.read_csv(
        FEATURES_PATH,
        low_memory=False,
    )

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    required_columns = (
        selected_features
        + [TARGET_COLUMN, GROUP_COLUMN, OBJECT_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nMissing required columns:")
        for column in missing_columns[:30]:
            print(f"  - {column}")

        raise ValueError(
            f"{len(missing_columns)} required columns are missing. "
            "The temporal dataset does not match the saved feature list. "
            "Do not substitute features automatically."
        )

    # --------------------------------------------------------
    # 5. Select the saved test trials
    # --------------------------------------------------------

    print("\n[5/8] Selecting held-out test trials...")

    df[GROUP_COLUMN] = df[GROUP_COLUMN].map(
        normalize_trial_id
    )

    test_df = df[
        df[GROUP_COLUMN].isin(test_trials)
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No test rows matched the saved trial IDs. "
            "Check the trial identifiers in the CSV and JSON."
        )

    found_trials = set(test_df[GROUP_COLUMN].dropna().unique())

    missing_trials = test_trials - found_trials

    if missing_trials:
        raise ValueError(
            f"{len(missing_trials)} test trials were not found in the "
            f"temporal dataset. Examples: {list(missing_trials)[:10]}"
        )

    if found_trials.intersection(train_trials):
        raise ValueError(
            "A training trial appeared in the selected test data."
        )

    labels = pd.to_numeric(
        test_df[TARGET_COLUMN],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError(
            "Missing or invalid target labels found in test data."
        )

    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(
            "The slip target must contain only binary labels 0 and 1."
        )

    y_true = labels.astype(int).to_numpy()

    # Preserve exact selected feature ordering.
    X_test = test_df[selected_features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X_test.isna().any().any():
        bad_columns = X_test.columns[
            X_test.isna().any()
        ].tolist()

        raise ValueError(
            "Missing, infinite, or non-numeric values were found in "
            "the selected features. Confirm the preprocessing used "
            "during training before proceeding.\n"
            f"Problematic columns: {bad_columns[:20]}"
        )

    print(f"Test windows:       {len(test_df):,}")
    print(f"Matched test trials: {len(found_trials)}")
    print(f"No-slip windows:    {np.sum(y_true == 0):,}")
    print(f"Slip windows:       {np.sum(y_true == 1):,}")

    # --------------------------------------------------------
    # 6. Load model and predict
    # --------------------------------------------------------

    print("\n[6/8] Loading existing CatBoost model...")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    # The selected model was trained with NumPy feature arrays.
    # Pass features in the saved selected_features.json order.
    X_array = X_test.to_numpy(dtype=np.float32)

    print("Predicting test-window probabilities...")

    probabilities = model.predict_proba(X_array)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    test_df["actual_slip"] = y_true
    test_df["slip_probability"] = probabilities
    test_df["predicted_slip"] = predictions

    # --------------------------------------------------------
    # 7. Overall metrics
    # --------------------------------------------------------

    print("\n[7/8] Calculating overall metrics...")

    overall_metrics = calculate_metrics(
        y_true,
        predictions,
        probabilities,
    )

    overall_metrics["decision_threshold"] = THRESHOLD
    overall_metrics["test_trials"] = int(
        test_df[GROUP_COLUMN].nunique()
    )
    overall_metrics["selected_feature_count"] = len(
        selected_features
    )

    save_json(
        overall_metrics,
        OUTPUT_DIR / "overall_metrics.json",
    )

    # --------------------------------------------------------
    # 8. Object-wise metrics
    # --------------------------------------------------------

    print("\n[8/8] Calculating object-wise and trial-wise metrics...")

    object_rows = []

    for object_name, group in test_df.groupby(
        OBJECT_COLUMN,
        dropna=False,
    ):
        metrics = calculate_metrics(
            group["actual_slip"],
            group["predicted_slip"],
            group["slip_probability"],
        )

        metrics["object"] = str(object_name)
        metrics["trials"] = int(
            group[GROUP_COLUMN].nunique()
        )

        object_rows.append(metrics)

    object_metrics = pd.DataFrame(object_rows)

    object_columns = [
        "object",
        "trials",
        "windows",
        "no_slip_windows",
        "slip_windows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    object_metrics = object_metrics[object_columns]

    object_metrics.to_csv(
        OUTPUT_DIR / "object_wise_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Trial-wise metrics
    # --------------------------------------------------------

    trial_rows = []

    for trial_id, group in test_df.groupby(
        GROUP_COLUMN,
        dropna=False,
    ):
        metrics = calculate_metrics(
            group["actual_slip"],
            group["predicted_slip"],
            group["slip_probability"],
        )

        object_values = (
            group[OBJECT_COLUMN]
            .dropna()
            .astype(str)
            .unique()
        )

        metrics["trial"] = str(trial_id)

        metrics["object"] = (
            object_values[0]
            if len(object_values) == 1
            else "UNKNOWN_OR_MULTIPLE"
        )

        trial_rows.append(metrics)

    trial_metrics = pd.DataFrame(trial_rows)

    trial_columns = [
        "trial",
        "object",
        "windows",
        "no_slip_windows",
        "slip_windows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    trial_metrics = trial_metrics[trial_columns]

    trial_metrics.to_csv(
        OUTPUT_DIR / "trial_wise_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save individual predictions
    # --------------------------------------------------------

    prediction_columns = [
        GROUP_COLUMN,
        OBJECT_COLUMN,
        TARGET_COLUMN,
        "actual_slip",
        "slip_probability",
        "predicted_slip",
    ]

    test_df[prediction_columns].to_csv(
        OUTPUT_DIR / "test_window_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("OVERALL TEST RESULTS")
    print("=" * 72)

    for key, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value:.6f}")
        else:
            print(f"{key:28s}: {value}")

    print("\n" + "=" * 72)
    print("OBJECT-WISE RESULTS")
    print("=" * 72)

    print(
        object_metrics[
            [
                "object",
                "trials",
                "windows",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 72)
    print("10 LOWEST-PERFORMING TRIALS BY F1")
    print("=" * 72)

    lowest_trials = trial_metrics.sort_values(
        ["f1", "windows"],
        ascending=[True, True],
    ).head(10)

    print(
        lowest_trials[
            [
                "trial",
                "object",
                "windows",
                "slip_windows",
                "no_slip_windows",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)

    print(f"\nResults directory:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. overall_metrics.json")
    print("2. object_wise_metrics.csv")
    print("3. trial_wise_metrics.csv")
    print("4. test_window_predictions.csv")


if __name__ == "__main__":
    main()