from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generating a large dataset with at least 1000 objects
X, y = make_classification(
    n_samples=1500, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Display dataset in table form
data_df = pd.DataFrame(
    data=X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
data_df['Target'] = y

print("Dataset:")
print(data_df.head())

# Initialize k-fold cross-validation
num_splits = 5
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform k-fold cross-validation
fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize Naive Bayes Classifier
    classifier = GaussianNB()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append scores to lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Output evaluation metrics for each fold
    print(f"\nEvaluation metrics for Fold {fold}:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}\n')

    # Apply PCA to visualize decision boundaries using the first two principal components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_train_pca, X_test_pca = X_pca[train_index], X_pca[test_index]

    # Fit classifier on PCA-transformed data for visualization
    classifier.fit(X_train_pca, y_train)

    # Plot decision boundaries and classifier for each fold
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                cmap='viridis', marker='o', edgecolors='k')
    plt.title(f'Decision Boundaries & Classifier - Fold {fold}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    h = 0.02  # step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    plt.show()

    fold += 1

# Calculate average scores across all folds
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

# Print average evaluation metrics
print("Average Evaluation Metrics:")
print(f'Average Accuracy: {avg_accuracy}')
print(f'Average Precision: {avg_precision}')
print(f'Average Recall: {avg_recall}')
print(f'Average F1-score: {avg_f1}')
