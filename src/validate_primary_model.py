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

warnings.filterwarnings("default")

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data_processed" / "slid_ch_features_temporal.csv"

FEATURE_PATH = (
    ROOT / "results" / "catboost_feature_selected"
    / "selected_features.json"
)

SPLIT_PATH = ROOT / "results" / "trial_split.json"

MODEL_PATH = (
    ROOT / "models" / "catboost_object_identity"
    / "catboost_object_identity.cbm"
)

SAVED_PREDICTIONS_PATH = (
    ROOT / "results" / "object_identity_experiment"
    / "object_aware_test_predictions.csv"
)

OUTPUT_DIR = ROOT / "results" / "primary_model_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.41, 0.50]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def load_features(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        features = data

    elif isinstance(data, dict):
        features = None

        for key in [
            "selected_features",
            "features",
            "feature_names",
        ]:
            if isinstance(data.get(key), list):
                features = data[key]
                break

        if features is None:
            raise ValueError(
                "Could not identify the selected feature list "
                f"in {path}. Keys: {list(data.keys())}"
            )
    else:
        raise ValueError("Unexpected feature JSON format.")

    if not features:
        raise ValueError("Selected feature list is empty.")

    if len(features) != len(set(features)):
        raise ValueError("Duplicate selected feature names found.")

    return features


def metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    predicted = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "windows": len(y_true),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(
            y_true, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_true, predicted, zero_division=0
        ),
        "f1": f1_score(
            y_true, predicted, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "pr_auc": (
            average_precision_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else np.nan
        ),
        "false_negative_rate": (
            fn / (fn + tp) if (fn + tp) else np.nan
        ),
    }


# ============================================================
# 3. CHECK FILES
# ============================================================

section("PRIMARY MODEL VALIDATION AUDIT")

required_files = [
    DATA_PATH,
    FEATURE_PATH,
    SPLIT_PATH,
    MODEL_PATH,
    SAVED_PREDICTIONS_PATH,
]

for path in required_files:
    print(f"{'OK' if path.exists() else 'MISSING'}: {path}")

missing_files = [str(p) for p in required_files if not p.exists()]

if missing_files:
    raise FileNotFoundError("\n".join(missing_files))


# ============================================================
# 4. LOAD FEATURES AND SPLIT
# ============================================================

selected_features = load_features(FEATURE_PATH)

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split_info = json.load(f)

train_trials = set(split_info["train_trials"])
test_trials = set(split_info["test_trials"])

overlap = train_trials.intersection(test_trials)

section("TRIAL SPLIT VALIDATION")

print(f"Training trials: {len(train_trials)}")
print(f"Testing trials:  {len(test_trials)}")
print(f"Shared trials:   {len(overlap)}")

if overlap:
    raise ValueError(
        f"Train/test overlap detected: {sorted(overlap)}"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

section("LOADING PROCESSED DATA")

df = pd.read_csv(DATA_PATH)
saved_df = pd.read_csv(SAVED_PREDICTIONS_PATH)

print(f"Feature dataset shape: {df.shape}")
print(f"Saved predictions shape: {saved_df.shape}")
print(f"Selected feature count: {len(selected_features)}")

required_columns = {"trial", "object", "slip"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing required columns: {required_columns - set(df.columns)}"
    )

if df[["trial", "object", "slip"]].isna().any().any():
    raise ValueError("Missing values in trial, object, or slip.")

df["slip"] = pd.to_numeric(df["slip"], errors="raise").astype(int)

if not set(df["slip"].unique()).issubset({0, 1}):
    raise ValueError("Slip labels must be 0 or 1.")

missing_features = [
    f for f in selected_features if f not in df.columns
]

if missing_features:
    raise ValueError(
        f"Selected features missing from dataset: {missing_features[:20]}"
    )


# ============================================================
# 6. CHECK LABEL-DERIVED FEATURES
# ============================================================

section("FEATURE LEAKAGE CHECK")

forbidden_terms = ["max_contact_speed", "slip_fraction"]

forbidden_features = [
    f for f in selected_features
    if any(term in f.lower() for term in forbidden_terms)
]

if forbidden_features:
    raise ValueError(
        f"Potential label leakage: {forbidden_features}"
    )

print("No explicitly forbidden label-derived features found.")
print(
    "This check does not prove that every possible source "
    "of data leakage has been eliminated."
)


# ============================================================
# 7. RECONSTRUCT TEST SET
# ============================================================

section("RECONSTRUCTING TEST SET")

test_df = df[df["trial"].isin(test_trials)].copy()

print(f"Test windows: {len(test_df):,}")
print(f"Unique test trials: {test_df['trial'].nunique()}")

missing_trials = test_trials - set(test_df["trial"])

if missing_trials:
    raise ValueError(
        f"Test trials missing from dataset: {sorted(missing_trials)}"
    )

if test_df.empty:
    raise ValueError("Test dataset is empty.")


# ============================================================
# 8. LOAD CATBOOST MODEL
# ============================================================

section("SAVED MODEL VALIDATION")

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

# CatBoostClassifier in this environment does not expose
# feature_count_ or get_feature_count().
# feature_names_ is used to inspect the loaded model.
model_feature_names = list(model.feature_names_)
model_feature_count = len(model_feature_names)

print(f"Model feature count: {model_feature_count}")
print(f"Selected IMU features: {len(selected_features)}")
print("Expected additional feature: object identity")

expected_count = len(selected_features) + 1

if model_feature_count != expected_count:
    raise ValueError(
        "Unexpected model feature count.\n"
        f"Expected {expected_count}; found {model_feature_count}."
    )

# If meaningful feature names are stored, check their order.
expected_names = selected_features + ["object"]

if model_feature_names and model_feature_names != expected_names:
    print(
        "WARNING: Stored model feature names do not exactly match "
        "the expected names/order. Reproducibility checks below "
        "will help identify a possible mismatch."
    )


# ============================================================
# 9. GENERATE FRESH PREDICTIONS
# ============================================================

section("GENERATING FRESH TEST PREDICTIONS")

input_columns = selected_features + ["object"]

X_test = test_df[input_columns].copy()
X_test["object"] = X_test["object"].astype(str)

y_test = test_df["slip"].to_numpy(dtype=int)

if X_test.isna().any().any():
    raise ValueError("Missing values found in model input features.")

probabilities = model.predict_proba(X_test)[:, 1]

if not np.isfinite(probabilities).all():
    raise ValueError("Non-finite prediction probabilities detected.")

if np.any((probabilities < 0) | (probabilities > 1)):
    raise ValueError("Predicted probabilities fall outside [0, 1].")

print("Predictions generated successfully.")
print(f"Number of predictions: {len(probabilities):,}")
print(f"Mean slip probability: {probabilities.mean():.6f}")


# ============================================================
# 10. OVERALL METRICS
# ============================================================

section("RECOMPUTED OVERALL TEST METRICS")

overall_rows = []

for threshold in THRESHOLDS:
    overall_rows.append(
        metrics(y_test, probabilities, threshold)
    )

overall_df = pd.DataFrame(overall_rows)

print(
    overall_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

overall_df.to_csv(
    OUTPUT_DIR / "recomputed_test_metrics.csv",
    index=False,
)


# ============================================================
# 11. OBJECT-WISE METRICS
# ============================================================

section("OBJECT-WISE TEST METRICS")

object_rows = []

objects = sorted(test_df["object"].astype(str).unique())

for object_name in objects:

    mask = (
        test_df["object"].astype(str).to_numpy() == object_name
    )

    y_object = y_test[mask]
    p_object = probabilities[mask]

    for threshold in THRESHOLDS:

        row = metrics(
            y_object,
            p_object,
            threshold,
        )

        row["object"] = object_name
        object_rows.append(row)

object_df = pd.DataFrame(object_rows)

columns_to_display = [
    "object",
    "threshold",
    "windows",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
    "false_negative_rate",
    "TN",
    "FP",
    "FN",
    "TP",
]

print(
    object_df[columns_to_display].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

object_df.to_csv(
    OUTPUT_DIR / "recomputed_object_wise_metrics.csv",
    index=False,
)


# ============================================================
# 12. COMPARE SAVED AND FRESH PREDICTIONS
# ============================================================

section("SAVED PREDICTION REPRODUCIBILITY")

print(f"Reconstructed test rows: {len(test_df):,}")
print(f"Saved prediction rows:  {len(saved_df):,}")

if len(saved_df) != len(test_df):

    print(
        "WARNING: Row counts differ. Direct row-by-row "
        "reproducibility cannot be verified."
    )

else:

    row_identity_matches = True

    for column in ["trial", "object", "slip"]:

        if column not in saved_df.columns:
            print(f"WARNING: Saved predictions lack '{column}'.")
            row_identity_matches = False
            continue

        saved_values = saved_df[column].astype(str).to_numpy()
        current_values = test_df[column].astype(str).to_numpy()

        matches = np.array_equal(saved_values, current_values)

        print(
            f"{column} row alignment: "
            f"{'PASS' if matches else 'MISMATCH'}"
        )

        if not matches:
            row_identity_matches = False

    probability_column = next(
        (
            col for col in [
                "slip_probability",
                "probability_slip",
                "prob_slip",
                "slip_prob",
            ]
            if col in saved_df.columns
        ),
        None,
    )

    if probability_column is None:

        print("No recognized saved probability column found.")

    elif row_identity_matches:

        saved_probabilities = pd.to_numeric(
            saved_df[probability_column],
            errors="coerce",
        ).to_numpy()

        if not np.isfinite(saved_probabilities).all():

            print(
                "WARNING: Saved probabilities contain "
                "missing or invalid values."
            )

        else:

            max_difference = float(
                np.max(
                    np.abs(
                        saved_probabilities - probabilities
                    )
                )
            )

            print(
                "Maximum absolute probability difference: "
                f"{max_difference:.12f}"
            )

            print(
                "Probability reproducibility: "
                + (
                    "PASS"
                    if max_difference < 1e-6
                    else "MISMATCH"
                )
            )


# ============================================================
# 13. SAVE FRESH PREDICTIONS
# ============================================================

section("SAVING FRESH PREDICTIONS")

fresh_df = test_df[
    ["trial", "object", "slip"]
].copy()

for time_column in ["time_start", "time_end"]:

    if time_column in test_df.columns:
        fresh_df[time_column] = test_df[time_column].to_numpy()

fresh_df["slip_probability"] = probabilities

for threshold in THRESHOLDS:

    fresh_df[f"prediction_at_{threshold:.2f}"] = (
        probabilities >= threshold
    ).astype(int)

fresh_path = OUTPUT_DIR / "fresh_test_predictions.csv"

fresh_df.to_csv(fresh_path, index=False)


# ============================================================
# 14. SAVE AUDIT SUMMARY
# ============================================================

summary = {
    "dataset": DATA_PATH.name,
    "model": MODEL_PATH.name,
    "selected_feature_count": len(selected_features),
    "model_feature_count": model_feature_count,
    "training_trial_count": len(train_trials),
    "testing_trial_count": len(test_trials),
    "train_test_trial_overlap": len(overlap),
    "test_window_count": len(test_df),
    "test_trial_count": int(test_df["trial"].nunique()),
    "objects": objects,
    "thresholds_evaluated": THRESHOLDS,
    "limitations": [
        "The existing test split has already been examined during development.",
        "The generated labels still require verification against the official methodology.",
        "Aggregate performance does not establish generalization to unseen objects.",
        "Pringles Can slip recall requires further investigation.",
    ],
}

summary_path = OUTPUT_DIR / "audit_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


# ============================================================
# 15. FINAL REPORT
# ============================================================

section("AUDIT COMPLETE")

print(f"Overall metrics: {OUTPUT_DIR / 'recomputed_test_metrics.csv'}")
print(f"Object-wise metrics: {OUTPUT_DIR / 'recomputed_object_wise_metrics.csv'}")
print(f"Fresh predictions: {fresh_path}")
print(f"Audit summary: {summary_path}")

print("\nThis is a reproducibility audit, not an independent validation.")
print("The original dataset and trained model were not modified.")