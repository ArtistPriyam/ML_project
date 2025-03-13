import pickle
import matplotlib.pyplot as plt

# Load processed data
with open("processed_data.pkl", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split training data for validation (80-20 split)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_split, y_train_split.values.ravel())

# Make predictions
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Evaluate the model
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))


with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

#plotting  confusion matrix 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Ensure your test set exists
# Example: imX_test, y_test should be defined before this step
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ensure `y_pred` is generated correctly
y_pred = model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create results directory if it doesn't exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Save the confusion matrix as an image
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

# Save the figure
save_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory

print(f"Confusion Matrix saved at: {save_path}")



# Save classification report
with open("results/model_performance.txt", "w") as f:
    f.write(classification_report(y_test, y_test_pred))

from sklearn.metrics import classification_report, accuracy_score

# Generate classification report
report = classification_report(y_test, y_pred)

# Save the report in the results folder
report_path = os.path.join(results_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

print(f"Classification Report saved at: {report_path}")

print(hasattr(model, "predict_proba"))  # Should return True
#saving roc curve 

