# -*- coding: utf-8 -*-
"""Fraud_Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PxQa1jyWuQzzU4HB0tRbw1_4buGffO1B
"""

from google.colab import files
uploaded = files.upload()

import zipfile
import os
import pandas as pd
from pathlib import Path

# Unzip uploaded file
with zipfile.ZipFile("data.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

# Find and load all .pkl files
pkl_files = list(Path("data").rglob("*.pkl"))
print(f"Found {len(pkl_files)} pickle files")

df = pd.concat([pd.read_pickle(str(f)) for f in pkl_files], ignore_index=True)

# Convert TX_DATETIME to datetime
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
df['TX_DAY'] = df['TX_DATETIME'].dt.day
df['TX_WEEKDAY'] = df['TX_DATETIME'].dt.weekday

# Encode IDs
from sklearn.preprocessing import LabelEncoder
df['CUSTOMER_ID_ENC'] = LabelEncoder().fit_transform(df['CUSTOMER_ID'])
df['TERMINAL_ID_ENC'] = LabelEncoder().fit_transform(df['TERMINAL_ID'])

# Drop NaNs (if any)
df = df.dropna()

# Define features and target
features = ['CUSTOMER_ID_ENC', 'TERMINAL_ID_ENC', 'TX_AMOUNT', 'TX_HOUR', 'TX_DAY', 'TX_WEEKDAY']
X = df[features]
y = df['TX_FRAUD']

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define models
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

models = {
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate models
results = {}

for name, model in models.items():
    try:
        print(f"\nTraining and evaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Use predict_proba only if available
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = None
            roc_auc = "N/A"

        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "confusion_matrix": confusion,
            "classification_report": report,
            "roc_auc_score": roc_auc
        }
    except Exception as e:
        print(f"Error training {name}: {e}")

# Display results
for name, result in results.items():
    print(f"\nModel: {name}")
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("Classification Report:")
    print(pd.DataFrame(result["classification_report"]).transpose())
    print("ROC AUC Score:", result["roc_auc_score"])
    print("-" * 60)

# Step 3: Recursively load all .pkl files
from pathlib import Path
import pandas as pd

# Find all .pkl files inside the 'data' folder and its subfolders
pkl_files = list(Path("data").rglob("*.pkl"))

print(f"Found {len(pkl_files)} pickle files")

# Load all DataFrames from pkl files and concatenate them
df = pd.concat([pd.read_pickle(str(f)) for f in pkl_files], ignore_index=True)

# Preview
df.head()

# Convert TX_DATETIME to datetime object
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

# Extract time-based features
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
df['TX_DAY'] = df['TX_DATETIME'].dt.day
df['TX_WEEKDAY'] = df['TX_DATETIME'].dt.weekday

# Encode CUSTOMER_ID and TERMINAL_ID
from sklearn.preprocessing import LabelEncoder

df['CUSTOMER_ID_ENC'] = LabelEncoder().fit_transform(df['CUSTOMER_ID'])
df['TERMINAL_ID_ENC'] = LabelEncoder().fit_transform(df['TERMINAL_ID'])

# Define features and label
features = ['CUSTOMER_ID_ENC', 'TERMINAL_ID_ENC', 'TX_AMOUNT', 'TX_HOUR', 'TX_DAY', 'TX_WEEKDAY']
X = df[features]
y = df['TX_FRAUD']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

models = {
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(random_state=42, max_iter=300),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "confusion_matrix": confusion,
        "classification_report": report,
        "roc_auc_score": roc_auc
    }

# Print results
for name, result in results.items():
    print(f"Model: {name}")
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("Classification Report:")
    print(pd.DataFrame(result["classification_report"]).transpose())
    print("ROC AUC Score:", result["roc_auc_score"])
    print("-" * 50)

