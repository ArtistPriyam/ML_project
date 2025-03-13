import os
import numpy as np
import pandas as pd

# Define the base path
base_path = "data_for_project/UCI_HAR_Dataset"

# Load features
features_path = os.path.join(base_path, "features.txt")
features = pd.read_csv(features_path, sep="\s+", header=None, names=["index", "feature"])

# Ensure unique feature names
feature_names = features["feature"].tolist()

# Load activity labels
activity_labels_path = os.path.join(base_path, "activity_labels.txt")
activity_labels = pd.read_csv(activity_labels_path, sep="\s+", header=None, names=["index", "activity"])

# Load training data
X_train_path = os.path.join(base_path, "train", "X_train.txt")
y_train_path = os.path.join(base_path, "train", "y_train.txt")
subject_train_path = os.path.join(base_path, "train", "subject_train.txt")

X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(y_train_path, header=None, names=["Activity"])
subject_train = pd.read_csv(subject_train_path, header=None, names=["Subject"])

# Load test data
X_test_path = os.path.join(base_path, "test", "X_test.txt")
y_test_path = os.path.join(base_path, "test", "y_test.txt")
subject_test_path = os.path.join(base_path, "test", "subject_test.txt")

X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(y_test_path, header=None, names=["Activity"])
subject_test = pd.read_csv(subject_test_path, header=None, names=["Subject"])

# Ensure feature names are unique before assigning
if len(feature_names) == X_train.shape[1] and len(feature_names) == X_test.shape[1]:
    X_train.columns = feature_names
    X_test.columns = feature_names
else:
    print("Error: Feature names count does not match data dimensions.")

# Display dataset information
print("Training Data Info:")
print(X_train.info())
print(y_train.info())
print(subject_train.info())

print("Test Data Info:")
print(X_test.info())
print(y_test.info())
print(subject_test.info())


# Check execution success
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Check for missing values
print("Missing values in training data:\n", X_train.isnull().sum().sum())
print("Missing values in test data:\n", X_test.isnull().sum().sum())

# Fill any missing values with column mean (if needed)
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Normalize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode activity labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train["Activity"])
y_test_encoded = encoder.transform(y_test["Activity"])

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
y_train = pd.DataFrame(y_train_encoded, columns=["Activity"])
y_test = pd.DataFrame(y_test_encoded, columns=["Activity"])

# Final check
print("Preprocessing completed!")
print(X_train.head())
print(y_train["Activity"].value_counts())

import pickle

# Save processed data
with open("processed_data.pkl", "wb") as f:
    pickle.dump((X_train, y_train, X_test, y_test), f)

print("Processed data saved successfully.")

