# train.py

import mlflow
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load preprocessed Titanic data
data = pd.read_csv('data/train_preprocessed.csv')

# Prepare feature and target variables
X = data.drop(columns=["Survived"])  # Remove target variable
y = data["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation metrics
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    return accuracy, precision, recall

# Define performance plot function
def plot_performance_metrics(accuracy, precision, recall, n_estimators, max_depth):
    metrics = ['Accuracy', 'Precision', 'Recall']
    values = [accuracy, precision, recall]
    
    fig = plt.figure(figsize=(8, 4))
    plt.title(f"n_estimators: {n_estimators}, max_depth: {max_depth}")
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.xlabel('Performance Metrics')
    plt.ylabel('Values')
    plt.ylim(0, 1)
    plt.savefig(f"RFR_n_estimators_{n_estimators}_max_depth_{max_depth}.png")
    plt.close(fig)
    
if __name__ == "__main__":
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    with mlflow.start_run():
        mlflow.sklearn.autolog()

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        accuracy, precision, recall = eval_metrics(y_test, clf.predict(X_test))
        plot_performance_metrics(accuracy, precision, recall, n_estimators, max_depth)
        
        mlflow.log_artifact(f"RFR_n_estimators_{n_estimators}_max_depth_{max_depth}.png")
