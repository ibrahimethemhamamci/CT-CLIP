
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import os

# replace with inferred csv
inferred = pd.read_csv('output/inferred.csv')
# replace with origimnal test csv
gt = pd.read_csv('./data/val.csv')

labels = inferred.columns[2:]
y_true = gt[labels]
y_pred = inferred[labels]

cm = multilabel_confusion_matrix(y_true, y_pred)
clf = classification_report(y_true, y_pred,target_names=labels)

# replace with saving directory
save_root = ''
with open(os.path.join(save_root, 'test_classification_report.txt'), 'w') as file:
  file.write(clf)


# manual calculation:

# Initialize lists to store metrics
precision_list = []
recall_list = []
f1_list = []
support_list = []

# Calculate metrics for each label
for matrix in cm:
    TN, FP, FN, TP = matrix.ravel()
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    support = TP + FN

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    support_list.append(support)

# Calculate weighted averages
total_support = np.sum(support_list)
weighted_precision = np.sum([precision * support for precision, support in zip(precision_list, support_list)]) / total_support
weighted_recall = np.sum([recall * support for recall, support in zip(recall_list, support_list)]) / total_support
weighted_f1 = np.sum([f1 * support for f1, support in zip(f1_list, support_list)]) / total_support

# Create a DataFrame to save metrics
metrics_df = pd.DataFrame({
    'Label': labels,
    'Precision': precision_list,
    'Recall': recall_list,
    'F1 Score': f1_list,
    'Support': support_list
})

# Add weighted averages as the last row
metrics_df.loc['Weighted Average'] = ['Weighted Average', weighted_precision, weighted_recall, weighted_f1, total_support]

# Save to CSV
metrics_df.to_csv(os.path.join(save_root, 'metrics_manual.csv'), index=False)



