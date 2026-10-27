from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "trial_validation"
    / "test_window_predictions.csv"
)

OUTPUT_DIR = ROOT / "results" / "trial_validation"

TARGET_OBJECT = "Pringles Can"

# ============================================================
# LOAD PREDICTIONS
# ============================================================

def main():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = [
        "trial",
        "object",
        "actual_slip",
        "slip_probability",
        "predicted_slip",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["actual_slip"] = df["actual_slip"].astype(int)
    df["predicted_slip"] = df["predicted_slip"].astype(int)
    df["slip_probability"] = pd.to_numeric(
        df["slip_probability"], errors="raise"
    )

    # ========================================================
    # 1. CLASS DISTRIBUTION BY OBJECT
    # ========================================================

    print("\n" + "=" * 75)
    print("1. ACTUAL AND PREDICTED CLASS DISTRIBUTION")
    print("=" * 75)

    distribution = (
        df.groupby("object")
        .agg(
            windows=("actual_slip", "size"),
            actual_slip=("actual_slip", "sum"),
            predicted_slip=("predicted_slip", "sum"),
        )
        .reset_index()
    )

    distribution["actual_no_slip"] = (
        distribution["windows"] - distribution["actual_slip"]
    )

    distribution["predicted_no_slip"] = (
        distribution["windows"] - distribution["predicted_slip"]
    )

    distribution["actual_slip_percentage"] = (
        100 * distribution["actual_slip"] / distribution["windows"]
    )

    distribution["predicted_slip_percentage"] = (
        100 * distribution["predicted_slip"] / distribution["windows"]
    )

    print(distribution.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    ))

    distribution.to_csv(
        OUTPUT_DIR / "diagnostic_class_distribution.csv",
        index=False,
    )

    # ========================================================
    # 2. TRIAL-WISE ERROR ANALYSIS
    # ========================================================

    print("\n" + "=" * 75)
    print("2. TRIAL-WISE ERROR ANALYSIS")
    print("=" * 75)

    trial_rows = []

    for (obj, trial), group in df.groupby(
        ["object", "trial"],
        dropna=False,
    ):

        y = group["actual_slip"].to_numpy()
        pred = group["predicted_slip"].to_numpy()
        prob = group["slip_probability"].to_numpy()

        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        no_slip_count = int((y == 0).sum())
        slip_count = int((y == 1).sum())

        trial_rows.append({
            "object": obj,
            "trial": trial,
            "windows": len(group),
            "actual_slip": slip_count,
            "actual_no_slip": no_slip_count,
            "predicted_slip": int(pred.sum()),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "false_positive_rate": (
                fp / no_slip_count if no_slip_count else np.nan
            ),
            "false_negative_rate": (
                fn / slip_count if slip_count else np.nan
            ),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(
                y, pred, zero_division=0
            ),
            "recall": recall_score(
                y, pred, zero_division=0
            ),
            "f1": f1_score(
                y, pred, zero_division=0
            ),
            "mean_probability": float(np.mean(prob)),
            "mean_probability_actual_slip": (
                float(np.mean(prob[y == 1]))
                if slip_count else np.nan
            ),
            "mean_probability_actual_no_slip": (
                float(np.mean(prob[y == 0]))
                if no_slip_count else np.nan
            ),
        })

    trial_df = pd.DataFrame(trial_rows)

    trial_df.to_csv(
        OUTPUT_DIR / "diagnostic_trial_errors.csv",
        index=False,
    )

    # Show the trials with the most false positives.
    print("\nTOP 10 TRIALS BY FALSE POSITIVES")

    print(
        trial_df.sort_values(
            "false_positives", ascending=False
        ).head(10)[
            [
                "object",
                "trial",
                "windows",
                "actual_slip",
                "actual_no_slip",
                "false_positives",
                "false_negatives",
                "false_positive_rate",
                "false_negative_rate",
                "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # 3. PRINGLES CAN INVESTIGATION
    # ========================================================

    print("\n" + "=" * 75)
    print("3. PRINGLES CAN DIAGNOSTICS")
    print("=" * 75)

    pringles = df[
        df["object"].astype(str).str.casefold()
        == TARGET_OBJECT.casefold()
    ].copy()

    if pringles.empty:
        raise ValueError(
            f"No rows found for object: {TARGET_OBJECT}"
        )

    y = pringles["actual_slip"].to_numpy()
    pred = pringles["predicted_slip"].to_numpy()
    prob = pringles["slip_probability"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    print(f"Windows: {len(pringles):,}")
    print(f"Actual slip: {int((y == 1).sum()):,}")
    print(f"Actual no-slip: {int((y == 0).sum()):,}")
    print(f"Predicted slip: {int((pred == 1).sum()):,}")
    print(f"Predicted no-slip: {int((pred == 0).sum()):,}")

    print(f"\nTrue negatives: {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives: {tp:,}")

    print(
        "\nMean probability for actual slip windows: "
        f"{np.mean(prob[y == 1]):.6f}"
    )

    print(
        "Mean probability for actual no-slip windows: "
        f"{np.mean(prob[y == 0]):.6f}"
    )

    print("\nPRINGLES CAN TRIAL SUMMARY")

    pringles_trials = trial_df[
        trial_df["object"].astype(str).str.casefold()
        == TARGET_OBJECT.casefold()
    ].sort_values(
        "false_positives", ascending=False
    )

    print(
        pringles_trials[
            [
                "trial",
                "windows",
                "actual_slip",
                "actual_no_slip",
                "predicted_slip",
                "false_positives",
                "false_negatives",
                "false_positive_rate",
                "false_negative_rate",
                "f1",
                "mean_probability_actual_no_slip",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # 4. THRESHOLD DIAGNOSTIC ONLY
    # ========================================================

    # This is diagnostic analysis, not threshold optimization.
    # Do not choose a new threshold based on these test results.

    print("\n" + "=" * 75)
    print("4. PRINGLES CAN: ERROR COUNTS AT SELECTED THRESHOLDS")
    print("=" * 75)

    print(
        "These are sensitivity checks only. Do not select a new "
        "threshold from the test set."
    )

    threshold_rows = []

    for threshold in [0.30, 0.35, 0.41, 0.45, 0.50, 0.60, 0.70]:

        threshold_pred = (
            pringles["slip_probability"].to_numpy() >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, threshold_pred, labels=[0, 1]
        ).ravel()

        threshold_rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y, threshold_pred),
            "precision": precision_score(
                y, threshold_pred, zero_division=0
            ),
            "recall": recall_score(
                y, threshold_pred, zero_division=0
            ),
            "f1": f1_score(
                y, threshold_pred, zero_division=0
            ),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        })

    threshold_df = pd.DataFrame(threshold_rows)

    print(
        threshold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    threshold_df.to_csv(
        OUTPUT_DIR / "diagnostic_pringles_thresholds.csv",
        index=False,
    )

    print("\nDiagnostic files saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()