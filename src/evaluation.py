import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a trained CNN model and the test data
# model: Your trained CNN model
# X_test: Your test features
# y_test: True labels for the test data

# Step 1: Make Predictions
y_pred = model.predict(X_test)  # Predictions for your test set
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# Step 2: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")

# Step 3: Calculate Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred_classes, average='weighted')  # weighted averages for multiclass
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 4: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 5: Detailed Classification Report
report = classification_report(y_test, y_pred_classes)
print("Classification Report:\n", report)

# If it's a binary classification, you can also compute ROC-AUC:
from sklearn.metrics import roc_auc_score

# If you're dealing with binary classification and have the probabilities of the positive class
# y_pred_prob = model.predict_proba(X_test)[:, 1]  # for binary classification
# roc_auc = roc_auc_score(y_test, y_pred_prob)
# print(f"ROC-AUC: {roc_auc:.4f}")
