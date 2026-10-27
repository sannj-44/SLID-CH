from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "catboost_progression_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RECORDED EXPERIMENT RESULTS
# Values are fractions, not percentages.
# No models are retrained by this script.
# ============================================================

data = [
    {
        "stage": 1,
        "experiment": "Baseline CatBoost",
        "features": 72,
        "threshold": 0.50,
        "accuracy": 0.8633,
        "precision": 0.9258,
        "recall": 0.8521,
        "f1": 0.8874,
        "roc_auc": 0.9465,
        "pr_auc": 0.9661,
    },
    {
        "stage": 2,
        "experiment": "Tuned CatBoost",
        "features": 72,
        "threshold": 0.50,
        "accuracy": 0.8646,
        "precision": 0.9260,
        "recall": 0.8541,
        "f1": 0.8886,
        "roc_auc": 0.9457,
        "pr_auc": 0.9651,
    },
    {
        "stage": 3,
        "experiment": "Temporal CatBoost",
        "features": 288,
        "threshold": 0.50,
        "accuracy": 0.878316,
        "precision": 0.929875,
        "recall": 0.873420,
        "f1": 0.900764,
        "roc_auc": 0.954982,
        "pr_auc": 0.970925,
    },
    {
        "stage": 4,
        "experiment": "Feature-Selected CatBoost",
        "features": 150,
        "threshold": 0.50,
        "accuracy": 0.879373,
        "precision": 0.932289,
        "recall": 0.872601,
        "f1": 0.901458,
        "roc_auc": 0.957078,
        "pr_auc": 0.972259,
    },
    {
        "stage": 5,
        "experiment": "Feature-Selected CatBoost",
        "features": 150,
        "threshold": 0.41,
        "accuracy": 0.889420,
        "precision": 0.921928,
        "recall": 0.901453,
        "f1": 0.911576,
        "roc_auc": 0.957078,
        "pr_auc": 0.972259,
    },
]

df = pd.DataFrame(data)


# ============================================================
# 1. SAVE COMPARISON TABLE
# ============================================================

print("=" * 80)
print("CATBOOST PERFORMANCE PROGRESSION")
print("=" * 80)

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

df["accuracy_gain_vs_baseline_pp"] = (
    df["accuracy"] - df.loc[0, "accuracy"]
) * 100

df["f1_gain_vs_baseline_pp"] = (
    df["f1"] - df.loc[0, "f1"]
) * 100

df.to_csv(
    OUTPUT_DIR / "catboost_progression.csv",
    index=False,
)

display_df = df[
    [
        "stage",
        "experiment",
        "features",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
].copy()

for metric in metrics:
    display_df[metric] = display_df[metric] * 100

print(
    display_df.to_string(
        index=False,
        formatters={
            metric: lambda value: f"{value:.2f}%"
            for metric in metrics
        },
    )
)


# ============================================================
# 2. ACCURACY IMPROVEMENT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["stage"],
    df["accuracy"] * 100,
    marker="o",
    linewidth=2,
)

for _, row in df.iterrows():
    plt.annotate(
        f'{row["accuracy"] * 100:.2f}%',
        (
            row["stage"],
            row["accuracy"] * 100,
        ),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )

plt.xticks(
    df["stage"],
    [
        "Baseline",
        "Tuned",
        "Temporal",
        "Feature\nselection",
        "Threshold\n0.41",
    ],
)

plt.xlabel("Experiment stage")
plt.ylabel("Accuracy (%)")
plt.title("CatBoost Accuracy Improvement Across Experiments")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "catboost_accuracy_progression.png",
    dpi=300,
)

plt.close()


# ============================================================
# 3. F1-SCORE IMPROVEMENT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["stage"],
    df["f1"] * 100,
    marker="o",
    linewidth=2,
)

for _, row in df.iterrows():
    plt.annotate(
        f'{row["f1"] * 100:.2f}%',
        (
            row["stage"],
            row["f1"] * 100,
        ),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )

plt.xticks(
    df["stage"],
    [
        "Baseline",
        "Tuned",
        "Temporal",
        "Feature\nselection",
        "Threshold\n0.41",
    ],
)

plt.xlabel("Experiment stage")
plt.ylabel("F1-score (%)")
plt.title("CatBoost F1-score Improvement Across Experiments")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "catboost_f1_progression.png",
    dpi=300,
)

plt.close()


# ============================================================
# 4. COMPARE ALL CLASSIFICATION METRICS
# ============================================================

plt.figure(figsize=(13, 7))

for metric in metrics:
    plt.plot(
        df["stage"],
        df[metric] * 100,
        marker="o",
        label=metric.upper(),
    )

plt.xticks(
    df["stage"],
    [
        "Baseline",
        "Tuned",
        "Temporal",
        "Feature\nselection",
        "Threshold\n0.41",
    ],
)

plt.xlabel("Experiment stage")
plt.ylabel("Score (%)")
plt.title("CatBoost: Comparison of Classification Metrics")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "catboost_all_metrics_progression.png",
    dpi=300,
)

plt.close()


# ============================================================
# 5. PRINT SUMMARY
# ============================================================

baseline = df.iloc[0]
best = df.iloc[-1]

accuracy_gain = (
    best["accuracy"] - baseline["accuracy"]
) * 100

f1_gain = (
    best["f1"] - baseline["f1"]
) * 100

recall_gain = (
    best["recall"] - baseline["recall"]
) * 100

precision_change = (
    best["precision"] - baseline["precision"]
) * 100

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Baseline accuracy: {baseline['accuracy'] * 100:.2f}%")
print(f"Final accuracy:    {best['accuracy'] * 100:.2f}%")
print(f"Accuracy gain:     {accuracy_gain:.2f} percentage points")

print()

print(f"Baseline F1-score: {baseline['f1'] * 100:.2f}%")
print(f"Final F1-score:    {best['f1'] * 100:.2f}%")
print(f"F1-score gain:     {f1_gain:.2f} percentage points")

print()

print(f"Recall change:     {recall_gain:+.2f} percentage points")
print(f"Precision change:  {precision_change:+.2f} percentage points")

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")

print(f"\nOutput directory:\n{OUTPUT_DIR}")