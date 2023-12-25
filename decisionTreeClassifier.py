from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Create a DataFrame to display the dataset
feature_names = breast_cancer.feature_names
data_df = pd.DataFrame(data=X, columns=feature_names)
data_df['Target'] = y

# Display the dataset in a table form
print("Dataset:")
print(data_df.head())

# Display dataset size
print(f"Number of samples: {len(X)}")
print(f"Number of features: {len(X[0])}")
print(f"Number of classes: {len(np.unique(y))}")

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

    # Initialize Decision Tree Classifier
    classifier = DecisionTreeClassifier()

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

    # Visualize Decision Tree for each fold
    plt.figure(figsize=(32, 18))
    plot_tree(classifier, filled=True, feature_names=feature_names,
              class_names=breast_cancer.target_names)
    plt.title(f'Decision Tree - Fold {fold}')
    plt.show()

    # Output evaluation metrics for each fold
    print(f"\nEvaluation metrics for Fold {fold}:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}\n')

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
