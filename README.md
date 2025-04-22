# Fraud Detection Using Gradient Boosting and Other ML Models

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![Pandas](https://img.shields.io/badge/pandas-1.3+-brightgreen)

A comprehensive fraud detection system that evaluates multiple machine learning models on transaction data, with Gradient Boosting as the primary algorithm.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Models Evaluated](#models-evaluated)
- [Performance Metrics](#performance-metrics)
- [Results Interpretation](#results-interpretation)

## Features

- Loads and processes transaction data from multiple pickle files
- Extracts temporal features from transaction timestamps
- Encodes categorical variables using Label Encoding
- Evaluates 8 different machine learning models:
  - Gradient Boosting (primary focus)
  - Random Forest
  - Logistic Regression
  - SVM
  - K-Nearest Neighbors
  - Neural Network
  - Decision Tree
  - Naive Bayes
- Provides comprehensive performance metrics including:
  - Confusion matrices
  - Classification reports (precision, recall, f1-score)
  - ROC AUC scores

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- (Optional) Google Colab environment for original notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Fraud-Detection-Using-GradientBoosting.git
cd Fraud-Detection-Using-GradientBoosting
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your transaction data files (in .pkl format) in a `data` directory
2. Run the script:
```python
python fraud_detection.py
```

## Data Preparation

The script expects transaction data with the following columns:
- `TX_DATETIME`: Transaction timestamp
- `CUSTOMER_ID`: Customer identifier
- `TERMINAL_ID`: Terminal identifier
- `TX_AMOUNT`: Transaction amount
- `TX_FRAUD`: Fraud label (0/1)

The data preparation pipeline:
1. Loads all .pkl files from the data directory
2. Converts timestamps to datetime objects
3. Extracts temporal features:
   - Hour of day
   - Day of month
   - Day of week
4. Encodes customer and terminal IDs
5. Splits data into train/test sets (80/20) with stratified sampling

## Models Evaluated

The system compares performance of these models:

| Model | Key Parameters |
|-------|----------------|
| Gradient Boosting | n_estimators=100, learning_rate=0.1, max_depth=5 |
| Random Forest | n_estimators=100 |
| Logistic Regression | max_iter=1000 |
| SVM | kernel='rbf' |
| K-Nearest Neighbors | default parameters |
| Neural Network | hidden_layer_sizes=(100,), max_iter=300 |
| Decision Tree | default parameters |
| Naive Bayes | Gaussian |

## Performance Metrics

For each model, the script calculates:
- Confusion matrix (TP, FP, TN, FN)
- Classification report:
  - Precision
  - Recall
  - F1-score
  - Support
- ROC AUC score (using predict_proba)

## Results Interpretation

The Gradient Boosting model typically performs best for fraud detection tasks due to:
- Built-in feature importance
- Handling of class imbalance
- Sequential learning from mistakes

Key metrics to evaluate:
1. **Recall** (True Positive Rate): Ability to catch actual fraud cases
2. **Precision**: Proportion of flagged cases that are actually fraudulent
3. **ROC AUC**: Overall discrimination ability between classes

Example output format for each model:
```
Model: Gradient Boosting
Confusion Matrix:
[[85290    11]
 [   30    92]]
Classification Report:
              precision    recall  f1-score   support
       False       1.00      1.00      1.00     85301
        True       0.89      0.75      0.82       122
    accuracy                           1.00     85423
   macro avg       0.95      0.88      0.91     85423
weighted avg       1.00      1.00      1.00     85423
ROC AUC Score: 0.987654321
'''
