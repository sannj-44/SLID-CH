from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "all_models_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RECORDED RESULTS FROM PREVIOUS EXPERIMENTS
#
# These are recorded experiment results, not newly trained models.
#
# SVC and KNN used a stratified 50,000-window training subset.
# The other listed models used the full training-window set.
# ============================================================

data = [
    {
        "model": "Logistic Regression",
        "family": "Linear",
        "training_data": "Full training set",
        "accuracy": 0.7217,
        "precision": 0.8246,
        "recall": 0.7111,
        "f1": 0.7637,
        "roc_auc": 0.8149,
        "pr_auc": 0.8563,
    },
    {
        "model": "Decision Tree",
        "family": "Decision Tree",
        "training_data": "Full training set",
        "accuracy": 0.7634,
        "precision": 0.8725,
        "recall": 0.7329,
        "f1": 0.7966,
        "roc_auc": 0.8066,
        "pr_auc": 0.8431,
    },
    {
        "model": "Random Forest",
        "family": "Ensemble Tree",
        "training_data": "Full training set",
        "accuracy": 0.8177,
        "precision": 0.9118,
        "recall": 0.7880,
        "f1": 0.8454,
        "roc_auc": 0.9242,
        "pr_auc": 0.9526,
    },
    {
        "model": "Extra Trees",
        "family": "Ensemble Tree",
        "training_data": "Full training set",
        "accuracy": 0.7626,
        "precision": 0.8467,
        "recall": 0.7627,
        "f1": 0.8025,
        "roc_auc": 0.8625,
        "pr_auc": 0.9092,
    },
    {
        "model": "SVM",
        "family": "Kernel",
        "training_data": "50,000-window subset",
        "accuracy": 0.7739,
        "precision": 0.8471,
        "recall": 0.7840,
        "f1": 0.8143,
        "roc_auc": 0.8425,
        "pr_auc": 0.8854,
    },
    {
        "model": "KNN",
        "family": "Instance-Based",
        "training_data": "50,000-window subset",
        "accuracy": 0.7468,
        "precision": 0.8815,
        "recall": 0.6927,
        "f1": 0.7758,
        "roc_auc": 0.8432,
        "pr_auc": 0.8835,
    },
    {
        "model": "XGBoost Baseline",
        "family": "Boosted Tree",
        "training_data": "Full training set",
        "accuracy": 0.8583,
        "precision": 0.9299,
        "recall": 0.8391,
        "f1": 0.8822,
        "roc_auc": 0.9475,
        "pr_auc": 0.9665,
    },
    {
        "model": "LightGBM",
        "family": "Boosted Tree",
        "training_data": "Full training set",
        "accuracy": 0.8547,
        "precision": 0.9215,
        "recall": 0.8420,
        "f1": 0.8799,
        "roc_auc": 0.9425,
        "pr_auc": 0.9638,
    },
    {
        "model": "CatBoost Baseline",
        "family": "Boosted Tree",
        "training_data": "Full training set",
        "accuracy": 0.8633,
        "precision": 0.9258,
        "recall": 0.8521,
        "f1": 0.8874,
        "roc_auc": 0.9465,
        "pr_auc": 0.9661,
    },
    {
        "model": "Tuned XGBoost",
        "family": "Boosted Tree",
        "training_data": "Full training set",
        "accuracy": 0.859628,
        "precision": 0.929503,
        "recall": 0.841847,
        "f1": 0.883506,
        "roc_auc": 0.948093,
        "pr_auc": 0.966859,
    },
    {
        "model": "Tuned CatBoost",
        "family": "Boosted Tree",
        "training_data": "Full training set",
        "accuracy": 0.8646,
        "precision": 0.9260,
        "recall": 0.8541,
        "f1": 0.8886,
        "roc_auc": 0.9457,
        "pr_auc": 0.9651,
    },
    {
        "model": "Temporal CatBoost",
        "family": "Boosted Tree + Temporal Features",
        "training_data": "Full training set",
        "accuracy": 0.878316,
        "precision": 0.929875,
        "recall": 0.873420,
        "f1": 0.900764,
        "roc_auc": 0.954982,
        "pr_auc": 0.970925,
    },
    {
        "model": "Feature-Selected CatBoost",
        "family": "Boosted Tree + Feature Selection",
        "training_data": "Full training set",
        "accuracy": 0.879373,
        "precision": 0.932289,
        "recall": 0.872601,
        "f1": 0.901458,
        "roc_auc": 0.957078,
        "pr_auc": 0.972259,
    },
    {
        "model": "Feature-Selected CatBoost (Threshold 0.41)",
        "family": "Boosted Tree + Feature Selection",
        "training_data": "Full training set",
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
# 1. SAVE COMPLETE RESULTS
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

df.to_csv(
    OUTPUT_DIR / "all_models_comparison.csv",
    index=False,
)

print("=" * 100)
print("ALL MACHINE LEARNING MODELS: COMPARISON")
print("=" * 100)

display_df = df.copy()

for metric in metrics:
    display_df[metric] = display_df[metric] * 100

print(
    display_df[
        ["model", "training_data"] + metrics
    ].to_string(
        index=False,
        formatters={
            metric: lambda value: f"{value:.2f}%"
            for metric in metrics
        },
    )
)


# ============================================================
# 2. RANK MODELS BY ACCURACY
# ============================================================

accuracy_ranking = df.sort_values(
    "accuracy",
    ascending=False,
).reset_index(drop=True)

accuracy_ranking.insert(
    0,
    "rank",
    range(1, len(accuracy_ranking) + 1),
)

accuracy_ranking.to_csv(
    OUTPUT_DIR / "models_ranked_by_accuracy.csv",
    index=False,
)

print("\n" + "=" * 100)
print("MODELS RANKED BY ACCURACY")
print("=" * 100)

print(
    accuracy_ranking[
        ["rank", "model", "accuracy", "f1"]
    ].to_string(
        index=False,
        formatters={
            "accuracy": lambda value: f"{value * 100:.2f}%",
            "f1": lambda value: f"{value * 100:.2f}%",
        },
    )
)


# ============================================================
# 3. ACCURACY COMPARISON GRAPH
# ============================================================

accuracy_sorted = df.sort_values(
    "accuracy",
    ascending=True,
)

plt.figure(figsize=(13, 9))

bars = plt.barh(
    accuracy_sorted["model"],
    accuracy_sorted["accuracy"] * 100,
)

for bar, value in zip(
    bars,
    accuracy_sorted["accuracy"] * 100,
):
    plt.text(
        value + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        va="center",
        fontsize=8,
    )

plt.xlabel("Accuracy (%)")
plt.ylabel("Model")
plt.title("Accuracy Comparison of All Trained ML Models")
plt.xlim(0, 103)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "all_models_accuracy.png",
    dpi=300,
)

plt.close()


# ============================================================
# 4. F1-SCORE COMPARISON GRAPH
# ============================================================

f1_sorted = df.sort_values(
    "f1",
    ascending=True,
)

plt.figure(figsize=(13, 9))

bars = plt.barh(
    f1_sorted["model"],
    f1_sorted["f1"] * 100,
)

for bar, value in zip(
    bars,
    f1_sorted["f1"] * 100,
):
    plt.text(
        value + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        va="center",
        fontsize=8,
    )

plt.xlabel("F1-score (%)")
plt.ylabel("Model")
plt.title("F1-score Comparison of All Trained ML Models")
plt.xlim(0, 103)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "all_models_f1.png",
    dpi=300,
)

plt.close()


# ============================================================
# 5. ROC-AUC AND PR-AUC COMPARISON
# ============================================================

plt.figure(figsize=(13, 8))

for metric in ["roc_auc", "pr_auc"]:
    plt.plot(
        df.sort_values("accuracy")["model"],
        df.sort_values("accuracy")[metric] * 100,
        marker="o",
        label=metric.upper(),
    )

plt.xticks(rotation=65, ha="right")
plt.ylabel("Score (%)")
plt.xlabel("Model")
plt.title("ROC-AUC and PR-AUC Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "all_models_auc.png",
    dpi=300,
)

plt.close()


# ============================================================
# 6. PRECISION, RECALL AND F1 COMPARISON
# ============================================================

top_models = df.sort_values(
    "f1",
    ascending=False,
).head(8)

x = range(len(top_models))
width = 0.25

plt.figure(figsize=(14, 7))

plt.bar(
    [i - width for i in x],
    top_models["precision"] * 100,
    width,
    label="Precision",
)

plt.bar(
    list(x),
    top_models["recall"] * 100,
    width,
    label="Recall",
)

plt.bar(
    [i + width for i in x],
    top_models["f1"] * 100,
    width,
    label="F1-score",
)

plt.xticks(
    list(x),
    top_models["model"],
    rotation=35,
    ha="right",
)

plt.ylabel("Score (%)")
plt.xlabel("Model")
plt.title("Precision, Recall and F1-score: Top 8 Models")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_models_precision_recall_f1.png",
    dpi=300,
)

plt.close()


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

best_accuracy = df.loc[df["accuracy"].idxmax()]
best_f1 = df.loc[df["f1"].idxmax()]
best_roc_auc = df.loc[df["roc_auc"].idxmax()]
best_pr_auc = df.loc[df["pr_auc"].idxmax()]

print("\n" + "=" * 100)
print("BEST RESULTS")
print("=" * 100)

print(
    f"Best accuracy: {best_accuracy['model']} "
    f"({best_accuracy['accuracy'] * 100:.2f}%)"
)

print(
    f"Best F1-score: {best_f1['model']} "
    f"({best_f1['f1'] * 100:.2f}%)"
)

print(
    f"Best ROC-AUC: {best_roc_auc['model']} "
    f"({best_roc_auc['roc_auc'] * 100:.2f}%)"
)

print(
    f"Best PR-AUC: {best_pr_auc['model']} "
    f"({best_pr_auc['pr_auc'] * 100:.2f}%)"
)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")

print(f"\nOutput directory:\n{OUTPUT_DIR}")


# ============================================================
# IMPORTANT INTERPRETATION NOTES
# ============================================================

print("\nIMPORTANT NOTES")
print("-" * 100)

print(
    "1. SVM and KNN used a 50,000-window training subset; "
    "their results are not directly comparable to models trained "
    "on all training windows."
)

print(
    "2. The final CatBoost result uses threshold 0.41, while most "
    "other results use threshold 0.50."
)

print(
    "3. The threshold was selected using out-of-fold training predictions, "
    "but the test set has been examined repeatedly."
)

print(
    "4. The results compare agreement with the existing derived labels. "
    "The label-generation methodology should be verified before making "
    "claims about real-world physical slip detection."
)

print(
    "5. CatBoost's highest reported accuracy does not establish that it "
    "will generalize best to previously unseen object types."
)