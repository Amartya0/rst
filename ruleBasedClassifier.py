import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
num_objects = 20
num_features = 3
data = np.random.randint(0, 3, size=(num_objects, num_features))
# Binary classification labels
labels = np.random.randint(0, 2, size=num_objects)

# Create a DataFrame to display the dataset
column_names = [f"Feature_{i}" for i in range(num_features)]
data_df = pd.DataFrame(data, columns=column_names)
data_df['Target'] = labels

# Display the dataset in a table form
print("Dataset:")
print(data_df)

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
for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Generate rules based on Decision Matrix Method
    rules = {}

    for feature in range(num_features):
        unique_values = np.unique(X_train[:, feature])
        unique_values.sort()

        # Create rules based on thresholds
        for value in unique_values:
            rule = f"If Feature_{feature} <= {value}, then label as {np.bincount(y_train[X_train[:, feature] <= value]).argmax()}"
            rules[f"Feature_{feature} <= {value}"] = rule

    # Apply rules to test set for prediction
    y_pred_rules = np.zeros_like(y_test)

    for rule in rules.values():
        feature_idx = int(rule.split()[1].split('_')[1])
        threshold = int((rule.split()[3])[0])
        mask = X_test[:, feature_idx] <= threshold
        y_pred_rules[mask] = int(rule.split()[-1])

    # Calculate evaluation metrics for rule-based predictions
    accuracy = accuracy_score(y_test, y_pred_rules)
    precision = precision_score(y_test, y_pred_rules)
    recall = recall_score(y_test, y_pred_rules)
    f1 = f1_score(y_test, y_pred_rules)

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
