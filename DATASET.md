# SLID-CH Dataset Documentation

## 1. Dataset Overview

SLID-CH (Slip Labels and Inertial Data from Compliant Hand) is the dataset used in this project for tactile slip detection using inertial measurement unit (IMU) signals.

The project investigates whether motion signals captured by IMU sensors can be used to identify object slipping during manipulation.

The dataset was created by Miranda Cravetz at Oregon State University and published on Zenodo on July 14, 2025.

**Official dataset record:** https://zenodo.org/records/15886336
**DOI:** https://doi.org/10.5281/zenodo.15886336

## 2. Dataset Characteristics

| Property                                | Description                                       |
| --------------------------------------- | ------------------------------------------------- |
| Dataset name                            | SLID-CH                                           |
| Full title                              | Slip Labels and Inertial Data from Compliant Hand |
| Creator                                 | Miranda Cravetz                                   |
| Affiliation                             | Oregon State University                           |
| Publication date                        | July 14, 2025                                     |
| Repository                              | Zenodo                                            |
| Application                             | Tactile slip detection                            |
| Primary input                           | IMU sensor time-series data                       |
| Number of original trials               | 195                                               |
| Retained trials after quality filtering | 192                                               |
| Reported raw timesteps                  | 721,392                                           |
| IMU channels                            | 12                                                |
| Approximate sampling rate               | 122 Hz                                            |
| Number of object categories             | 5                                                 |
| Window duration                         | 250 ms                                            |
| Window overlap                          | 50%                                               |
| Statistical features                    | 84                                                |
| Learning task                           | Slip versus non-slip classification               |

The trial count, timestep count, sampling configuration, windowing parameters, and feature count describe the configuration reported for this project. They should be checked against the final preprocessing outputs before publication.

## 3. Object Categories

The five object categories used in this project are:

1. Plastic Cup
2. Pringles Can
3. Pudding Box
4. Rubik's Cube
5. Soup Can

These are the object categories used in the project's experiments. Confirm their correspondence with the original dataset labels before publication.

## 4. Data Processing

The preprocessing workflow includes:

1. Inspecting the original dataset and labels.
2. Identifying and excluding faulty trials.
3. Preparing the IMU signals and associated labels.
4. Applying contact-related masking where applicable.
5. Dividing the time-series signals into windows.
6. Extracting statistical features from the IMU channels.
7. Preparing the resulting features for machine-learning experiments.

The reported preprocessing configuration retains 192 of the 195 original trials.

The preprocessing implementation is available in `src/prepare_dataset.py`.

## 5. Feature Engineering

The primary feature representation contains 84 statistical features, derived from 12 IMU channels using seven statistics per channel:

* Mean
* Standard deviation
* Variance
* Minimum
* Maximum
* Range
* Root mean square (RMS)

The project also investigates additional approaches, including temporal feature engineering, feature selection, threshold optimization, and object-identity information.

These experiments evaluate whether alternative feature representations and modeling strategies improve slip detection performance.

## 6. Labeling and Slip Detection

Optical-marker kinematics were used in this project to establish slip-related labels, with a reported slip threshold of approximately 0.06 m/s.

The exact labeling procedure, coordinate definitions, threshold implementation, and label distribution should be verified against the final preprocessing code and the original dataset documentation.

The classification task distinguishes slip from non-slip observations.

## 7. Experimental Methodology

The project compares multiple machine-learning approaches, including:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Extra Trees
* Support Vector Classifier (SVC)
* LightGBM
* XGBoost
* CatBoost
* Soft-voting ensemble methods

The evaluation uses trial-grouped validation to reduce data leakage between windows derived from the same physical trial.

The reported methodology uses five-fold GroupKFold validation grouped by trial ID. Feature scaling, where required, should be fitted using training data only.

The experiments also investigate object-identity information and per-object performance. This is important because strong aggregate classification metrics do not necessarily indicate reliable slip detection for every object category.

Results, comparison tables, classification reports, confusion matrices, and feature-importance outputs are organized in the `results/` directory.

## 8. Dataset Access, Citation, and License

### Official Dataset Source

**Title:** Slip Labels and Inertial Data from Compliant Hand (SLID-CH)

**Creator:** Miranda Cravetz

**Affiliation:** Oregon State University

**Publication date:** July 14, 2025

**Repository:** Zenodo

**DOI:** 10.5281/zenodo.15886336

**Official record:** https://zenodo.org/records/15886336

The dataset re
