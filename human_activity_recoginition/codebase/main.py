import os
import pickle
import pandas as pd

# Load the model
model_path = "models/logistic_model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load new sensor data
new_data_path = "data_for_project/new_sensor_data.csv"
new_data = pd.read_csv(new_data_path)

# Preprocess the data (if needed)
features_path = "data_for_project/UCI_HAR_Dataset/features.txt"
features = pd.read_csv(features_path, sep="\s+", header=None, names=["index", "feature"])
feature_names = features["feature"].tolist()

if len(feature_names) == new_data.shape[1]:
    new_data.columns = feature_names

# Make predictions
y_pred = model.predict(new_data)

# Map the activity labels
activity_labels_path = "data_for_project/UCI_HAR_Dataset/activity_labels.txt"
activity_labels = pd.read_csv(activity_labels_path, sep="\s+", header=None, names=["id", "activity"])
activity_mapping = dict(zip(activity_labels["id"], activity_labels["activity"]))

# Convert predictions to activity names
y_pred_named = [activity_mapping.get(label, "Unknown") for label in y_pred]

# Save predictions to results folder
results_path = "results/predictions.csv"
pd.DataFrame(y_pred_named, columns=["Predicted Activity"]).to_csv(results_path, index=False)

print("Predictions saved to", results_path)








