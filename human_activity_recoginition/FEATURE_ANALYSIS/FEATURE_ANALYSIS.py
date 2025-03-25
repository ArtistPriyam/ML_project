import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#----------------------------------------------------------------------
#STATISTICAL ANALYSIS (FEATURE VARIABILITY AND DISTRIBUTION)
#--------------------------------------------------------------

# Load feature names
features = pd.read_csv("data_base/UCI_HAR_Dataset/features.txt", sep="\s+", header=None, names=["index", "feature"])

# Load dataset
X_train = pd.read_csv("Data_base/UCI_HAR_Dataset/train/X_train.txt", sep="\s+", header=None)

# Compute variance of each feature
variances = X_train.var(axis=0)

# Sort and visualize top 20 features with highest variance
top_variance_indices = np.argsort(variances)[-20:]
plt.figure(figsize=(15, 5))
plt.barh(features.iloc[top_variance_indices]["feature"], variances.iloc[top_variance_indices])
plt.xlabel("Variance")
plt.ylabel("Feature")
plt.title("Top 20 Most Variable Features")
result_dir ="results/feature_analysis_result"
save_path = os.path.join(result_dir, "feature_analysis_variance.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory
print(f"file saved at saved at: {save_path}")

#---------------------------------------------
# FEATURE IMPORTANCE USING RANDOM FOREST
#---------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
# Load labels
y_train = pd.read_csv("data_base/UCI_HAR_Dataset/train/Y_train.txt", header=None).values.ravel()

# Train a RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Sort features by importance
top_feature_indices = np.argsort(importances)[-20:]

# Print and visualize
print("Top 10 Important Features:")
for i in top_feature_indices:
    print(features.iloc[i]["feature"], ":", importances[i])

plt.figure(figsize=(15, 10))
plt.barh(features.iloc[top_feature_indices]["feature"], importances[top_feature_indices])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features in HAR Dataset")
result_dir ="results/feature_analysis_result"
save_path = os.path.join(result_dir, "feature_analysis_IMP_F_20.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory
print(f"file saved at saved at: {save_path}")

#--------------------------------------------------------------
# DIMENSIONALITY REDUCTION USING PCA 
#-------------------------------------
from sklearn.decomposition import PCA

# Normalize data before PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_train_scaled)

# Number of principal components selected
print(f"Reduced to {X_pca.shape[1]} features from {X_train.shape[1]}")

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Explained Variance vs. Number of Components")
plt.grid()
save_path = os.path.join(result_dir, "dimension_reduction.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory
print(f"file saved at saved at: {save_path}")


#--------------------------------------------------------
#t_SNE for data visualisation
from sklearn.manifold import TSNE

# Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_train_scaled)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap="viridis", alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of HAR Data")
plt.colorbar(label="Activity Class")
save_path = os.path.join(result_dir, "T_SNE.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory
print(f"file saved at saved at: {save_path}")