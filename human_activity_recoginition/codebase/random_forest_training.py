import pickle
import matplotlib.pyplot as plt

# Load processed data
with open("processed_data.pkl", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split training data for validation (80-20 split)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Train a Logistic Regression model
model =  RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split.values.ravel())

# Make predictions
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Evaluate the model
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))


with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

#"______________________________________________________________"
#"PLOTTING RESULTS "

#[1] plotting  confusion matrix 
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Ensure `y_pred` is generated correctly
y_pred = model.predict(X_test)
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Create results directory if it doesn't exist
results_dir = "results/result_random_forest_model"
os.makedirs(results_dir, exist_ok=True)
# Save the confusion matrix as an image
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)  #sklearn.metrices
disp.plot(cmap=plt.cm.Blues)

# Save the figure
save_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory
print(f"Confusion Matrix saved at: {save_path}")

# Save classification report--------------------------------------------
from sklearn.metrics import classification_report, accuracy_score
# Generate classification report
report = classification_report(y_test, y_pred)
# Save the report in the results folder
report_path = os.path.join(results_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification Report saved at: {report_path}")

#saving accuracy ------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
with open("results/result_random_forest_model/model_performance.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n") #f-string formatting to directly store variable inside the string 

#ROC curve -------------------------------------------------------------

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Binarize the labels for multi-class
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
n_classes = y_test_bin.shape[1]

# Predict probabilities
y_prob = model.predict_proba(X_test)

# Plotting ROC curve
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc='lower right')
plt.grid()
save_path = os.path.join(results_dir, "roc_curve.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory
print(f"roc_curve saved at: {save_path}")