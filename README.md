# SLID-CH: IMU-Based Tactile Slip Detection

A machine-learning project investigating slip detection using inertial measurement unit (IMU) signals from the SLID-CH dataset.

## Overview

This project explores the classification of slip and non-slip events using IMU time-series data. It includes dataset preprocessing, statistical feature extraction, machine-learning model training, model comparison, threshold optimization, and trial-grouped validation.

The goal is to evaluate different machine-learning approaches and investigate their ability to identify slip events across multiple object categories.

## Dataset

**Dataset:** Slip Labels and Inertial Data from Compliant Hand (SLID-CH)
**Creator:** Miranda Cravetz
**Affiliation:** Oregon State University
**Publication date:** July 14, 2025
**DOI:** [10.5281/zenodo.15886336](https://doi.org/10.5281/zenodo.15886336)
**Official source:** [Zenodo dataset record](https://zenodo.org/records/15886336)

The dataset contains inertial measurements and associated information used for slip detection research.

### Dataset Configuration Used in This Project

| Property                                | Value             |
| --------------------------------------- | ----------------- |
| Original trials                         | 195               |
| Trials retained after quality filtering | 192               |
| Reported raw timesteps                  | 721,392           |
| IMU channels                            | 12                |
| Approximate sampling frequency          | 122 Hz            |
| Object categories                       | 5                 |
| Window duration                         | 250 ms            |
| Window overlap                          | 50%               |
| Statistical features                    | 84                |
| Classification task                     | Slip vs. non-slip |

These values describe the reported project configuration and should be verified against the final preprocessing outputs.

### Object Categories

* Plastic Cup
* Pringles Can
* Pudding Box
* Rubik's Cube
* Soup Can

### Dataset Access

The original dataset is not included in this repository. Download it from the official [SLID-CH Zenodo record](https://zenodo.org/records/15886336).

See [`DATASET.md`](DATASET.md) for dataset details, attribution, licensing information, and reproduction notes.

## Methodology

### 1. Data Preprocessing

* Inspect dataset files and labels.
* Identify and exclude faulty trials.
* Prepare IMU signals and associated labels.
* Apply contact-related masking where applicable.
* Divide time-series signals into overlapping windows.

### 2. Feature Engineering

The primary feature representation uses 84 statistical features derived from 12 IMU channels:

* Mean
* Standard deviation
* Variance
* Minimum
* Maximum
* Range
* Root mean square (RMS)

Additional experiments investigate temporal features, feature selection, and object-identity information.

### 3. Machine-Learning Models

The project includes experiments with:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Extra Trees
* Support Vector Classifier (SVC)
* LightGBM
* XGBoost
* CatBoost
* Soft-voting ensembles

### 4. Evaluation

The reported evaluation uses five-fold GroupKFold validation grouped by trial ID to reduce leakage between windows from the same physical trial.

The project also investigates model tuning, classification thresholds, feature importance, and performance across object categories.

**Note:** High overall accuracy does not necessarily indicate reliable detection for every object category. Per-object metrics and slip-class performance should also be examined.

## Repository Structure

```text
SLID-CH/
├── src/                   # Preprocessing, feature engineering, training and evaluation
├── results/               # Metrics, figures and experiment outputs
├── reports/               # Dataset quality and descriptive statistics
├── DATASET.md             # Dataset documentation and citation
├── README.md              # Project overview and setup instructions
├── requirements.txt       # Python dependencies
├── SLID-CH_Project_Report review2.pdf
├── SLID_CH_Final_Comprehensive_ML_Notebook.ipynb
├── SLID_CH_Final_Transparent_Classical_ML_Notebook.ipynb
└── SLID_CH_Tactile_Slip_Detection_Final_Review_Deck.pptx
```

The original dataset, processed data, trained model checkpoints, and local development settings are excluded from the repository.

## Installation

### Prerequisites

* Python version compatible with the project dependencies
* Git
* The original SLID-CH dataset

### 1. Clone the repository

```bash
git clone https://github.com/sannj-44/SLID-CH.git
cd SLID-CH
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Ubuntu/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Obtain the dataset

Download the dataset from the official Zenodo record and extract it according to the input paths expected by the preprocessing scripts.

### 5. Run the pipeline

Review `src/prepare_dataset.py` and the selected training script before execution. Update local data paths if necessary.

The exact commands depend on each script's arguments and configuration.

## Results

The `results/` directory contains available experiment outputs, including:

* Model comparison tables and plots
* Accuracy, precision, recall, F1-score, and other metrics where reported
* Confusion matrices
* Feature-importance analyses
* Hyperparameter tuning results
* Threshold optimization results
* Trial-level and object-level evaluation summaries

Consult the final project report and notebooks for the interpretation of these results.

## Reproducibility

Reproducing the experiments requires:

1. Access to the original dataset.
2. Compatible Python and library versions.
3. The preprocessing and training scripts.
4. The correct data paths and experiment configuration.
5. Consistent trial splits and random seeds where applicable.

The results in this repository are provided for reference. Exact reproduction should be verified by rerunning the pipeline.

## Dataset Attribution and License

Cravetz, M. (2025). *Slip Labels and Inertial Data from Compliant Hand (SLID-CH).* Zenodo. https://doi.org/10.5281/zenodo.15886336

Consult the official dataset record for the applicable license and redistribution conditions before redistributing dataset files or derivatives.

## Limitations

* Results depend on preprocessing, dataset quality, trial splits, and model configurations.
* Aggregate metrics may conceal poor performance on particular object categories.
* Object identity may improve classification metrics without establishing generalization to unseen objects.
* The original dataset and trained model checkpoints are not included in this repository.

## Acknowledgments

This project uses the SLID-CH dataset created by Miranda Cravetz at Oregon State University.

Dataset source: https://zenodo.org/records/15886336
