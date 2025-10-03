# Network Intrusion Detection using Machine Learning and Deep Learning

This project implements and evaluates various machine learning and deep learning models for network intrusion detection using the CIC-IDS2017 dataset. It addresses the task from two perspectives:
1.  **Multi-Class Classification:** Classifying network traffic into benign or specific attack categories.
2.  **Anomaly Detection:** Distinguishing between normal (benign) and anomalous (attack) traffic.

The notebook covers the entire pipeline, from data loading and extensive preprocessing to model training, evaluation, and comparison.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Preprocessing & Cleaning](#1-data-preprocessing--cleaning)
  - [2. Handling Class Imbalance](#2-handling-class-imbalance)
  - [3. Multi-Class Classification](#3-multi-class-classification)
  - [4. Anomaly Detection](#4-anomaly-detection)
- [Results](#results)
  - [Multi-Class Classification Performance](#multi-class-classification-performance)
  - [Anomaly Detection Performance](#anomaly-detection-performance)
- [Setup and Usage](#setup-and-usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [File Structure](#file-structure)

## Features
- **Data Integration:** Loads and merges data from all eight CIC-IDS2017 CSV files.
- **Comprehensive Preprocessing:** Includes cleaning column names, handling duplicates, infinite values, and NaNs.
- **Attack Categorization:** Groups 14 specific attack labels into 8 broader, more manageable categories.
- **Imbalance Handling:** Utilizes a hybrid approach of Random Undersampling for the majority class and SMOTE (Synthetic Minority Over-sampling Technique) for minority classes.
- **Diverse Model Implementation:**
    - **Multi-Class Classification:** Logistic Regression, Decision Tree, Random Forest, LightGBM, and a soft-voting Ensemble Classifier.
    - **Deep Learning for Classification:** A hybrid CNN-LSTM model.
    - **Anomaly Detection:** Isolation Forest, Local Outlier Factor (LOF), Autoencoder, and a 1D-CNN.
- **Performance Evaluation:** Detailed performance metrics including accuracy, precision, recall, F1-score, and classification reports for all models.

## Dataset
The project uses the **CIC-IDS2017 dataset**, which contains benign network traffic and a variety of up-to-date common attacks, resembling true real-world data (PCAPs).

- **Source:** The data is compiled from network traffic captured over 5 days.
- **Characteristics:** The raw dataset consists of over 2.8 million records and 79 features. It is highly imbalanced, with the 'BENIGN' class constituting the vast majority of the data.

The original 15 labels are mapped into 8 main categories for classification:
`BENIGN`, `DDoS`, `DoS`, `PortScan`, `Brute Force`, `Bot`, `Web Attack`, `Infiltration`, and `Heartbleed`.

## Methodology

### 1. Data Preprocessing & Cleaning
The raw data is first loaded and concatenated into a single Pandas DataFrame. The following steps are then performed:
1.  **Duplicate Removal:** Duplicate rows are dropped.
2.  **Handling Invalid Values:** Infinite values (`np.inf`, `-np.inf`) are replaced with `NaN`, and all rows containing `NaN` values are subsequently dropped.
3.  **Attack Consolidation:** Similar attack types are grouped under a common parent category (e.g., 'DoS Hulk', 'DoS GoldenEye' are mapped to 'DoS').
4.  **Label Encoding:** The categorical target variable ('Attack Type') is converted to numerical labels using `sklearn.preprocessing.LabelEncoder`.
5.  **Feature Scaling:** All numerical features are scaled using `sklearn.preprocessing.StandardScaler` to normalize the data and improve model performance.
6.  **Data Splitting:** The data is split into training (70%) and testing (30%) sets, stratified by the target variable to maintain class distribution.

### 2. Handling Class Imbalance
The dataset is highly imbalanced, which can bias models towards the majority class. To mitigate this, a hybrid resampling strategy is applied **only to the training data**:
- **RandomUnderSampler:** The majority 'BENIGN' class is down-sampled to 200,000 instances.
- **SMOTE (Over-sampling):** All minority attack classes are synthetically over-sampled to match the new majority class count (200,000 instances).

This results in a perfectly balanced training set for the multi-class classification models.

### 3. Multi-Class Classification
The goal is to classify each network flow into one of the 8 attack categories or as benign.
- **Traditional ML Models:**
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - LightGBM Classifier
- **Ensemble Model:** A `VotingClassifier` is created using the best-performing models (Logistic Regression, Random Forest, LightGBM) with soft voting to leverage their collective predictive power.
- **Deep Learning Model (CNN-LSTM):** A hybrid model is built using TensorFlow/Keras. The architecture is as follows:
    1.  `Conv1D` layer to extract spatial features from the input.
    2.  `MaxPooling1D` layer.
    3.  `LSTM` layer to capture temporal dependencies in the sequence.
    4.  `Dense` layers for final classification.
    The model is trained using the `sparse_categorical_crossentropy` loss function.

### 4. Anomaly Detection
The goal is to classify traffic simply as 'normal' (1) or 'anomaly' (-1). For this binary task, models are trained to identify any deviation from benign traffic.
- **Unsupervised Models:**
    - **Isolation Forest:** An ensemble method that explicitly identifies anomalies by isolating them.
    - **Local Outlier Factor (LOF):** A density-based algorithm that identifies outliers by measuring the local deviation of a given data point with respect to its neighbors. The model is trained only on benign data and `novelty=True` is used for prediction.
- **Deep Learning Models:**
    - **Autoencoder:** A neural network trained only on benign data to reconstruct its input. A high reconstruction error (MSE) on new data suggests it is an anomaly. The anomaly threshold is set at the 95th percentile of the reconstruction error on the benign test set.
    - **1D-CNN:** A time-series approach where a 1D Convolutional Neural Network is trained as a binary classifier, also using only benign data for training.

## Results

### Multi-Class Classification Performance
The models trained on the resampled data achieved excellent performance, with tree-based ensembles leading the way.

| Model                 | Accuracy |
| --------------------- | :------: |
| Random Forest         | **99.85%** |
| Decision Tree         | 99.84%   |
| Voting Classifier     | 99.29%   |
| LightGBM              | 96.69%   |
| Logistic Regression   | 85.40%   |
| CNN-LSTM              | ~99.9% (Val Acc) |

### Anomaly Detection Performance
The models were evaluated on their ability to detect any type of attack traffic. The Local Outlier Factor demonstrated the best balance of precision and recall for identifying anomalies.

| Model                | Accuracy | 
| -------------------- | :------: | 
| **Local Outlier Factor** | **86.78%** | 
| Autoencoder          | 83.71%   | 
| Isolation Forest     | 82.31%   | 
| 1D-CNN               | 83.18%   | 


## Setup and Usage

### Prerequisites
The project is built using Python 3.11. The required libraries are listed below:
- `tensorflow`
- `keras-tuner`
- `imbalanced-learn`
- `lightgbm`
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`

### Installation
1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

### Running the Project
1.  **Dataset:** Download the CIC-IDS2017 dataset CSV files and place them in a folder named `CICIDS2017` in the root of the project directory.
2.  **Jupyter Notebook:** Open and run the cells in the `Network Intrusion Detection System (NIDS).ipynb` notebook. The notebook is self-contained and will execute all the steps from data loading to model evaluation.

## File Structure
```
├── CICIDS2017/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   └── ... (and other dataset CSV files)
├── Classification and detection.ipynb
└── README.md 
```