from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SLID-CH : DATA PROCESSING + EDA
# ============================================================
#
# This script demonstrates:
#
# DATA CLEANING
#   1. Missing-value detection
#   2. Missing-value handling
#   3. Invalid-value checking
#   4. Duplicate checking
#
# DATA REDUCTION
#   5. Duplicate removal
#
# DATA TRANSFORMATION
#   6. Feature transformation
#   7. Min-Max normalization
#   8. Z-score standardization
#
# DISCRETIZATION
#   9. Continuous -> categorical bins
#
# STATISTICAL ANALYSIS
#   10. Mean
#   11. Median
#   12. Mode
#   13. MAD
#   14. Variance
#   15. Standard deviation
#   16. Quartile deviation
#   17. Skewness
#   18. Kurtosis
#
# VISUALIZATION
#   19. 2x2 dataset dashboard
#   20. Central tendency
#   21. Histograms
#   22. Frequency polygon
#   23. Box plots
#   24. Scatter plots
#   25. Correlation heatmap
#   26. Transformation comparison
#
# IMPORTANT:
# IMUSlip/ is never modified.
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data_processed/slid_ch_features.csv"
)

DATA_DIR = Path("data_processed")
FIGURE_DIR = Path("figures")
REPORT_DIR = Path("reports")

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("SLID-CH DATA PROCESSING + EDA")
print("=" * 80)

print("\n[1] Loading processed feature dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

TARGET = "slip"

META_COLUMNS = [
    "object",
    "trial"
]

LEAKAGE_COLUMNS = [
    "max_contact_speed",
    "slip_fraction"
]

NUMERIC_COLUMNS = df.select_dtypes(
    include=np.number
).columns.tolist()

FEATURE_COLUMNS = [
    column
    for column in NUMERIC_COLUMNS
    if column not in [
        TARGET,
        *LEAKAGE_COLUMNS
    ]
]

print("\nColumn information:")
print(f"Numerical columns      : {len(NUMERIC_COLUMNS)}")
print(f"ML candidate features  : {len(FEATURE_COLUMNS)}")
print(f"Target                 : {TARGET}")
print(f"Metadata               : {META_COLUMNS}")
print(f"Analysis-only columns  : {LEAKAGE_COLUMNS}")


# ============================================================
# DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 80)
print("2. DATA UNDERSTANDING")
print("=" * 80)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nObject distribution:")
print(df["object"].value_counts())


# ============================================================
# 3. MISSING VALUE DETECTION
# ============================================================

print("\n" + "=" * 80)
print("3. MISSING VALUE DETECTION")
print("=" * 80)

missing_count = df.isna().sum()

missing_percentage = (
    df.isna().mean() * 100
)

missing_report = pd.DataFrame({
    "feature": df.columns,
    "missing_count": missing_count.values,
    "missing_percentage": missing_percentage.values
})

missing_report.to_csv(
    REPORT_DIR / "missing_values.csv",
    index=False
)

total_missing = int(
    df.isna().sum().sum()
)

print(
    f"\nTotal missing values: {total_missing:,}"
)

if total_missing == 0:
    print("Result: No missing values detected.")
else:
    print("\nColumns containing missing values:")
    print(
        missing_report[
            missing_report["missing_count"] > 0
        ].to_string(index=False)
    )


# ============================================================
# 4. MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 80)
print("4. MISSING VALUE HANDLING")
print("=" * 80)

df_clean = df.copy()

before_missing = int(
    df_clean.isna().sum().sum()
)

if before_missing > 0:

    # Numerical features -> median imputation
    for column in FEATURE_COLUMNS:

        if df_clean[column].isna().any():

            median_value = df_clean[column].median()

            df_clean[column] = (
                df_clean[column].fillna(median_value)
            )

    # Numeric analysis columns
    for column in LEAKAGE_COLUMNS:

        if column in df_clean.columns:
            if df_clean[column].isna().any():

                median_value = df_clean[column].median()

                df_clean[column] = (
                    df_clean[column].fillna(median_value)
                )

    after_missing = int(
        df_clean.isna().sum().sum()
    )

    print(
        f"Missing values before handling : {before_missing:,}"
    )

    print(
        f"Missing values after handling  : {after_missing:,}"
    )

else:

    print("No missing values required imputation.")
    print("Dataset retained without artificial modification.")


# ============================================================
# 5. INVALID VALUE CHECK
# ============================================================

print("\n" + "=" * 80)
print("5. INVALID VALUE CHECK")
print("=" * 80)

invalid_rows = []

for column in NUMERIC_COLUMNS:

    if column not in df_clean.columns:
        continue

    nan_count = int(
        df_clean[column].isna().sum()
    )

    inf_count = int(
        np.isinf(
            df_clean[column].to_numpy()
        ).sum()
    )

    invalid_rows.append({
        "feature": column,
        "NaN_count": nan_count,
        "infinite_count": inf_count,
        "total_invalid": nan_count + inf_count
    })


invalid_report = pd.DataFrame(
    invalid_rows
)

invalid_report.to_csv(
    REPORT_DIR / "invalid_values.csv",
    index=False
)

total_invalid = int(
    invalid_report["total_invalid"].sum()
)

print(
    f"\nTotal invalid numerical values: {total_invalid:,}"
)

if total_invalid == 0:
    print("Result: All numerical values are finite and valid.")
else:
    print(
        invalid_report[
            invalid_report["total_invalid"] > 0
        ].to_string(index=False)
    )


# ============================================================
# 6. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("6. DUPLICATE CHECK")
print("=" * 80)

duplicate_count = int(
    df_clean.duplicated().sum()
)

rows_before = len(df_clean)

print(
    f"Rows before duplicate reduction : {rows_before:,}"
)

print(
    f"Exact duplicate rows             : {duplicate_count:,}"
)

if duplicate_count > 0:

    df_clean = (
        df_clean
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Rows after duplicate reduction  : {len(df_clean):,}"
    )

else:

    print(
        "No exact duplicate rows found."
    )

    print(
        f"Rows after duplicate reduction  : {len(df_clean):,}"
    )


# ============================================================
# 7. DUPLICATE REPORT
# ============================================================

duplicate_report = pd.DataFrame({
    "measure": [
        "Rows before reduction",
        "Duplicate rows",
        "Rows after reduction",
        "Rows removed"
    ],
    "count": [
        rows_before,
        duplicate_count,
        len(df_clean),
        rows_before - len(df_clean)
    ]
})

duplicate_report.to_csv(
    REPORT_DIR / "duplicate_analysis.csv",
    index=False
)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

df_clean.to_csv(
    DATA_DIR / "cleaned_dataset.csv",
    index=False
)

print(
    "\nCleaned dataset saved:"
)

print(
    DATA_DIR / "cleaned_dataset.csv"
)


# ============================================================
# 8. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("7. DESCRIPTIVE STATISTICS")
print("=" * 80)

descriptive_statistics = (
    df_clean[FEATURE_COLUMNS]
    .describe()
    .T
)

descriptive_statistics.to_csv(
    REPORT_DIR / "descriptive_statistics.csv"
)

print(
    descriptive_statistics
    .round(4)
    .to_string()
)


# ============================================================
# 9. CENTRAL TENDENCY + DISPERSION
# ============================================================

print("\n" + "=" * 80)
print("8. CENTRAL TENDENCY + DISPERSION")
print("=" * 80)

STAT_FEATURES = [
    "left_ax_mean",
    "left_ax_std",
    "left_gx_mean",
    "left_gx_std",
    "right_ax_mean",
    "right_ax_std",
    "right_gx_mean",
    "right_gx_std",
    "max_contact_speed"
]

STAT_FEATURES = [
    column
    for column in STAT_FEATURES
    if column in df_clean.columns
]


central_rows = []

for column in STAT_FEATURES:

    values = df_clean[column].dropna()

    mean_value = values.mean()
    median_value = values.median()

    mode_values = values.mode()

    if len(mode_values) > 0:
        mode_value = mode_values.iloc[0]
    else:
        mode_value = np.nan

    # Mean Absolute Deviation
    mad = np.mean(
        np.abs(values - mean_value)
    )

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    quartile_deviation = (
        q3 - q1
    ) / 2

    variance = values.var()
    standard_deviation = values.std()

    skewness = values.skew()
    kurtosis = values.kurtosis()

    central_rows.append({
        "feature": column,
        "mean": mean_value,
        "median": median_value,
        "mode": mode_value,
        "MAD": mad,
        "Q1": q1,
        "Q3": q3,
        "quartile_deviation": quartile_deviation,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "skewness": skewness,
        "kurtosis": kurtosis
    })


central_tendency = pd.DataFrame(
    central_rows
)

central_tendency.to_csv(
    REPORT_DIR / "central_tendency.csv",
    index=False
)

print(
    central_tendency
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 10. OUTLIER ANALYSIS — IQR
# ============================================================

print("\n" + "=" * 80)
print("9. OUTLIER ANALYSIS USING IQR")
print("=" * 80)

outlier_rows = []

for column in FEATURE_COLUMNS:

    values = df_clean[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (values < lower_bound)
        |
        (values > upper_bound)
    )

    count = int(
        outlier_mask.sum()
    )

    percentage = (
        count / len(values)
    ) * 100

    outlier_rows.append({
        "feature": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": count,
        "outlier_percentage": percentage
    })


outlier_report = pd.DataFrame(
    outlier_rows
)

outlier_report.to_csv(
    REPORT_DIR / "outlier_analysis.csv",
    index=False
)

print(
    outlier_report
    .sort_values(
        "outlier_count",
        ascending=False
    )
    .head(20)
    .round(4)
    .to_string(index=False)
)

print(
    "\nImportant: IQR outliers are identified for analysis."
)

print(
    "They are NOT automatically deleted because extreme IMU"
    " measurements may represent genuine slip-related motion."
)


# ============================================================
# 11. MIN-MAX NORMALIZATION
# ============================================================

print("\n" + "=" * 80)
print("10. MIN-MAX NORMALIZATION")
print("=" * 80)

normalized_df = df_clean.copy()

for column in FEATURE_COLUMNS:

    minimum = normalized_df[column].min()
    maximum = normalized_df[column].max()

    if maximum != minimum:

        normalized_df[column] = (
            normalized_df[column] - minimum
        ) / (
            maximum - minimum
        )

    else:

        normalized_df[column] = 0.0


normalized_df.to_csv(
    DATA_DIR / "normalized_dataset.csv",
    index=False
)

print(
    "\nNormalization completed."
)

print(
    "Expected range: [0, 1]"
)

normalization_check = pd.DataFrame({
    "minimum": normalized_df[FEATURE_COLUMNS].min(),
    "maximum": normalized_df[FEATURE_COLUMNS].max()
})

print(
    normalization_check
    .round(4)
    .head(15)
    .to_string()
)


# ============================================================
# 12. Z-SCORE STANDARDIZATION
# ============================================================

print("\n" + "=" * 80)
print("11. Z-SCORE STANDARDIZATION")
print("=" * 80)

standardized_df = df_clean.copy()

for column in FEATURE_COLUMNS:

    mean_value = standardized_df[column].mean()
    std_value = standardized_df[column].std()

    if std_value != 0:

        standardized_df[column] = (
            standardized_df[column] - mean_value
        ) / std_value

    else:

        standardized_df[column] = 0.0


standardization_statistics = pd.DataFrame({
    "original_mean":
        df_clean[FEATURE_COLUMNS].mean(),

    "original_std":
        df_clean[FEATURE_COLUMNS].std(),

    "standardized_mean":
        standardized_df[FEATURE_COLUMNS].mean(),

    "standardized_std":
        standardized_df[FEATURE_COLUMNS].std()
})

standardization_statistics.to_csv(
    REPORT_DIR / "transformation_statistics.csv"
)

print(
    standardization_statistics
    .round(4)
    .head(15)
    .to_string()
)


# ============================================================
# 13. DISCRETIZATION
# ============================================================

print("\n" + "=" * 80)
print("12. DATA DISCRETIZATION")
print("=" * 80)

discretized_df = df_clean.copy()


# Contact-speed bins
speed_bins = [
    -np.inf,
    0.02,
    0.06,
    np.inf
]

speed_labels = [
    "Low",
    "Medium",
    "High"
]

discretized_df[
    "contact_speed_category"
] = pd.cut(
    discretized_df["max_contact_speed"],
    bins=speed_bins,
    labels=speed_labels,
    include_lowest=True
)


# Slip-fraction bins
slip_fraction_bins = [
    -0.001,
    0.25,
    0.50,
    0.75,
    1.001
]

slip_fraction_labels = [
    "Very Low",
    "Low",
    "High",
    "Very High"
]

discretized_df[
    "slip_fraction_category"
] = pd.cut(
    discretized_df["slip_fraction"],
    bins=slip_fraction_bins,
    labels=slip_fraction_labels,
    include_lowest=True
)


discretized_df.to_csv(
    DATA_DIR / "discretized_dataset.csv",
    index=False
)

print(
    "\nContact-speed categories:"
)

print(
    discretized_df[
        "contact_speed_category"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nSlip-fraction categories:"
)

print(
    discretized_df[
        "slip_fraction_category"
    ].value_counts(
        dropna=False
    )
)


# ============================================================
# VISUALIZATION
# ============================================================


# ------------------------------------------------------------
# Common samples
# ------------------------------------------------------------

sample = df_clean.sample(
    n=min(30000, len(df_clean)),
    random_state=42
)


# ============================================================
# FIGURE 01 — 2x2 DATASET DASHBOARD
# ============================================================

print("\n" + "=" * 80)
print("13. VISUALIZATION — DATASET DASHBOARD")
print("=" * 80)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig.suptitle(
    "SLID-CH Dataset Overview",
    fontsize=16,
    fontweight="bold"
)


# BAR CHART
class_counts = (
    df_clean["slip"]
    .value_counts()
    .sort_index()
)

axes[0, 0].bar(
    ["No Slip", "Slip"],
    class_counts.values
)

axes[0, 0].set_title(
    "Slip / No-Slip Window Count"
)

axes[0, 0].set_ylabel(
    "Number of Windows"
)

for i, value in enumerate(
    class_counts.values
):

    axes[0, 0].text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom"
    )


# PIE CHART
axes[0, 1].pie(
    class_counts.values,
    labels=["No Slip", "Slip"],
    autopct="%1.1f%%"
)

axes[0, 1].set_title(
    "Class Proportion"
)


# HISTOGRAM
axes[1, 0].hist(
    sample["left_ax_std"],
    bins=40
)

axes[1, 0].set_title(
    "Distribution of Left ax Variability"
)

axes[1, 0].set_xlabel(
    "Left ax Standard Deviation"
)

axes[1, 0].set_ylabel(
    "Frequency"
)


# BOX PLOT
no_slip_values = df_clean.loc[
    df_clean["slip"] == 0,
    "left_ax_std"
].sample(
    n=min(
        20000,
        int((df_clean["slip"] == 0).sum())
    ),
    random_state=42
)

slip_values = df_clean.loc[
    df_clean["slip"] == 1,
    "left_ax_std"
].sample(
    n=min(
        20000,
        int((df_clean["slip"] == 1).sum())
    ),
    random_state=42
)

axes[1, 1].boxplot(
    [
        no_slip_values,
        slip_values
    ],
    labels=["No Slip", "Slip"]
)

axes[1, 1].set_title(
    "Left ax Variability by Class"
)

axes[1, 1].set_ylabel(
    "Standard Deviation"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_dataset_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 02 — CENTRAL TENDENCY
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig.suptitle(
    "Central Tendency of Selected IMU Features",
    fontsize=16,
    fontweight="bold"
)

central_plot_features = [
    "left_ax_mean",
    "left_ax_std",
    "left_gx_mean",
    "left_gx_std"
]

for ax, column in zip(
    axes.flat,
    central_plot_features
):

    values = df_clean[column]

    plot_values = values.sample(
        n=min(30000, len(values)),
        random_state=42
    )

    ax.hist(
        plot_values,
        bins=40
    )

    ax.axvline(
        values.mean(),
        linestyle="--",
        label="Mean"
    )

    ax.axvline(
        values.median(),
        linestyle=":",
        label="Median"
    )

    ax.set_title(column)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_central_tendency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 03 — HISTOGRAMS
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig.suptitle(
    "IMU Feature Distributions",
    fontsize=16,
    fontweight="bold"
)

hist_features = [
    "left_ax_mean",
    "left_ay_std",
    "right_ax_mean",
    "right_ay_std"
]

for ax, column in zip(
    axes.flat,
    hist_features
):

    ax.hist(
        sample[column],
        bins=50
    )

    ax.set_title(column)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_histograms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 04 — FREQUENCY POLYGON
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

feature = "left_ax_std"

no_slip = df_clean.loc[
    df_clean["slip"] == 0,
    feature
].sample(
    n=min(
        20000,
        int((df_clean["slip"] == 0).sum())
    ),
    random_state=42
)

slip = df_clean.loc[
    df_clean["slip"] == 1,
    feature
].sample(
    n=min(
        20000,
        int((df_clean["slip"] == 1).sum())
    ),
    random_state=42
)

bins = np.linspace(
    min(
        no_slip.min(),
        slip.min()
    ),
    max(
        no_slip.max(),
        slip.max()
    ),
    40
)

no_slip_hist, edges = np.histogram(
    no_slip,
    bins=bins
)

slip_hist, _ = np.histogram(
    slip,
    bins=bins
)

centres = (
    edges[:-1] + edges[1:]
) / 2

ax.plot(
    centres,
    no_slip_hist,
    marker="o",
    markersize=3,
    label="No Slip"
)

ax.plot(
    centres,
    slip_hist,
    marker="o",
    markersize=3,
    label="Slip"
)

ax.set_title(
    "Frequency Polygon: Slip vs No Slip"
)

ax.set_xlabel(
    "Left ax Standard Deviation"
)

ax.set_ylabel(
    "Frequency"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_frequency_polygon.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 05 — BOXPLOTS
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig.suptitle(
    "IMU Feature Spread: Slip vs No Slip",
    fontsize=16,
    fontweight="bold"
)

box_features = [
    "left_ax_std",
    "left_gx_std",
    "right_ax_std",
    "right_gx_std"
]

for ax, column in zip(
    axes.flat,
    box_features
):

    no_slip = df_clean.loc[
        df_clean["slip"] == 0,
        column
    ].sample(
        n=min(
            20000,
            int((df_clean["slip"] == 0).sum())
        ),
        random_state=42
    )

    slip = df_clean.loc[
        df_clean["slip"] == 1,
        column
    ].sample(
        n=min(
            20000,
            int((df_clean["slip"] == 1).sum())
        ),
        random_state=42
    )

    ax.boxplot(
        [no_slip, slip],
        labels=["No Slip", "Slip"]
    )

    ax.set_title(column)
    ax.set_ylabel("Value")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 06 — SCATTER PLOTS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

scatter_sample = df_clean.sample(
    n=min(15000, len(df_clean)),
    random_state=42
)


axes[0].scatter(
    scatter_sample["left_ax_std"],
    scatter_sample["left_gx_std"],
    alpha=0.35,
    s=8
)

axes[0].set_title(
    "Acceleration vs Gyroscope Variability"
)

axes[0].set_xlabel(
    "Left ax Standard Deviation"
)

axes[0].set_ylabel(
    "Left gx Standard Deviation"
)


axes[1].scatter(
    scatter_sample["max_contact_speed"],
    scatter_sample["left_ax_std"],
    alpha=0.35,
    s=8
)

axes[1].axvline(
    0.06,
    linestyle="--",
    label="Slip threshold"
)

axes[1].set_title(
    "Contact Speed vs IMU Variability"
)

axes[1].set_xlabel(
    "Maximum Contact Speed"
)

axes[1].set_ylabel(
    "Left ax Standard Deviation"
)

axes[1].legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 07 — CORRELATION HEATMAP
# ============================================================

correlation_features = [
    "left_ax_mean",
    "left_ax_std",
    "left_ay_std",
    "left_az_std",
    "left_gx_mean",
    "left_gx_std",
    "left_gy_std",
    "left_gz_std",
    "right_ax_mean",
    "right_ax_std",
    "right_ay_std",
    "right_az_std",
    "right_gx_mean",
    "right_gx_std",
    "right_gy_std",
    "right_gz_std",
    "max_contact_speed",
    "slip"
]

correlation_features = [
    column
    for column in correlation_features
    if column in df_clean.columns
]

correlation_matrix = (
    df_clean[
        correlation_features
    ].corr()
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

image = ax.imshow(
    correlation_matrix,
    aspect="auto"
)

ax.set_xticks(
    range(len(correlation_features))
)

ax.set_yticks(
    range(len(correlation_features))
)

ax.set_xticklabels(
    correlation_features,
    rotation=90
)

ax.set_yticklabels(
    correlation_features
)

ax.set_title(
    "Correlation Matrix of Selected Features"
)

fig.colorbar(
    image,
    ax=ax,
    label="Correlation"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 08 — TRANSFORMATION COMPARISON
# ============================================================

feature = "left_ax_std"

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

plot_original = df_clean[
    feature
].sample(
    n=min(30000, len(df_clean)),
    random_state=42
)

plot_normalized = normalized_df[
    feature
].sample(
    n=min(30000, len(normalized_df)),
    random_state=42
)

plot_standardized = standardized_df[
    feature
].sample(
    n=min(30000, len(standardized_df)),
    random_state=42
)


axes[0].hist(
    plot_original,
    bins=50
)

axes[0].set_title(
    "Original"
)

axes[0].set_xlabel(
    feature
)

axes[1].hist(
    plot_normalized,
    bins=50
)

axes[1].set_title(
    "Min-Max Normalized"
)

axes[1].set_xlabel(
    "Normalized Value"
)

axes[2].hist(
    plot_standardized,
    bins=50
)

axes[2].set_title(
    "Z-Score Standardized"
)

axes[2].set_xlabel(
    "Z-score"
)

fig.suptitle(
    "Effect of Data Transformation",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "08_transformation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DATA PROCESSING + EDA COMPLETE")
print("=" * 80)

print(
    f"\nOriginal rows       : {len(df):,}"
)

print(
    f"Final cleaned rows  : {len(df_clean):,}"
)

print(
    f"Duplicate rows removed : {duplicate_count:,}"
)

print(
    f"Missing values after cleaning : "
    f"{int(df_clean.isna().sum().sum()):,}"
)

print(
    f"ML candidate features : {len(FEATURE_COLUMNS)}"
)

print("\nGenerated data files:")

for file in sorted(
    DATA_DIR.glob("*.csv")
):
    print(" ", file)

print("\nGenerated report files:")

for file in sorted(
    REPORT_DIR.glob("*.csv")
):
    print(" ", file)

print("\nGenerated figure files:")

for file in sorted(
    FIGURE_DIR.glob("*.png")
):
    print(" ", file)

print("\n" + "=" * 80)