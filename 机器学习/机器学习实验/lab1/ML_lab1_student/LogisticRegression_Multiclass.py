import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1-score for predictions.

    Args:
    y_true (array): Actual true labels.
    y_pred (array): Predicted labels.

    Returns:
    dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = 0.
    precision = 0.
    recall = 0.
    f1 = 0.

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def manual_metrics_calculation(y_true, y_pred):
    """
    Manually calculate accuracy, precision, recall, and F1-score for predictions.

    Args:
    y_true (array): Actual true labels.
    y_pred (array): Predicted labels.

    Returns:
    dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = 0.
    precision = 0.
    recall = 0.
    f1 = 0.

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets


# Initialize and fit the logistic regression model


# Make predictions


# Calculate metrics using the encapsulated function


# Calculate metrics manually


# Plotting


# Plotting training data

