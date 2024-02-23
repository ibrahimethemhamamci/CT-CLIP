import numpy as np
import pandas as pd
import torch
import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import math
from eval import evaluate_internal

def sigmoid(tensor):
    """
    Computes sigmoid activation function.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying sigmoid function.
    """
    return 1 / (1 + torch.exp(-tensor))

def find_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities.
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if current_roc <= best_roc:
            best_roc = current_roc
            best_threshold = threshold

    return best_threshold

# Dictionary mapping labels to diagnoses
label_to_diagnosis = {
    0: 'Medical material',
    1: 'Arterial wall calcification',
    2: 'Cardiomegaly',
    3: 'Pericardial effusion',
    4: 'Coronary artery wall calcification',
    5: 'Hiatal hernia',
    6: 'Lymphadenopathy',
    7: 'Emphysema',
    8: 'Atelectasis',
    9: 'Lung nodule',
    10: 'Lung opacity',
    11: 'Pulmonary fibrotic sequela',
    12: 'Pleural effusion',
    13: 'Mosaic attenuation pattern',
    14: 'Peribronchial thickening',
    15: 'Consolidation',
    16: 'Bronchiectasis',
    17: 'Interlobular septal thickening',
}

# Path to inference output directory
data_dir = "/path_to_inference_output/"

# Load predicted and labels data
predicted_data = np.load(Path(data_dir) / 'predicted_weights.npz')
labels_data = np.load(Path(data_dir) / 'labels_weights.npz')

# Extracting the arrays from the loaded files
labels = labels_data['data']
predicted = predicted_data['data']

# Thresholds list
thresholds = []

# Find threshold for each label
for i in range(18):
    logit = predicted[:, i]
    l = labels[:, i]
    prob = logit
    threshold = find_threshold(prob, l)
    thresholds.append(threshold)

# Initialize DataFrames for storing evaluation metrics
concatenated_df_auroc = pd.DataFrame()
concatenated_df_f1 = pd.DataFrame()
concatenated_df_acc = pd.DataFrame()
concatenated_df_precision = pd.DataFrame()

# Bootstrap iterations
for _ in tqdm.tqdm(range(1000)):
    # Sampled data
    indices = np.random.choice(range(len(labels)), size=len(labels), replace=True)
    #sampled_labels = labels[indices]
    #sampled_predicted = predicted[indices]
    sampled_labels = labels
    sampled_predicted = predicted
    # Pathologies list
    pathologies = ['Medical material', 'Calcification', 'Cardiomegaly', 'Pericardial effusion',
                   'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy',
                   'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
                   'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
                   'Bronchiectasis', 'Interlobular septal thickening']

    # Evaluate internal metrics
    dfs_auroc = evaluate_internal(sampled_predicted, sampled_labels, pathologies, data_dir)
    concatenated_df_auroc = pd.concat([concatenated_df_auroc, dfs_auroc])

    # Write AUROC to Excel
    writer = pd.ExcelWriter(Path(data_dir) / 'aurocs_bootstrap.xlsx', engine='xlsxwriter')
    concatenated_df_auroc.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()

    f1s = []
    accs = []
    precisions = []

    # Calculate metrics for each label
    for i in range(18):
        prob = sampled_predicted[:, i]
        label = sampled_labels[:, i]
        threshold = thresholds[i]
        pred = (prob > threshold).astype(int)

        f1 = f1_score(label, pred, average="weighted")
        acc = accuracy_score(label, pred)
        precision = precision_score(label, pred)

        f1s.append(f1)
        accs.append(acc)
        precisions.append(precision)

    # Store metrics in DataFrames
    dfs_f1 = pd.DataFrame([f1s], columns=pathologies)
    dfs_acc = pd.DataFrame([accs], columns=pathologies)
    dfs_precision = pd.DataFrame([precisions], columns=pathologies)

    concatenated_df_f1 = pd.concat([concatenated_df_f1, dfs_f1])
    concatenated_df_acc = pd.concat([concatenated_df_acc, dfs_acc])
    concatenated_df_precision = pd.concat([concatenated_df_precision, dfs_precision])

    # Write F1, accuracy, and precision to Excel
    writer = pd.ExcelWriter(Path(data_dir) / 'f1_bootstrap.xlsx', engine='xlsxwriter')
    concatenated_df_f1.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()

    writer = pd.ExcelWriter(Path(data_dir) / 'acc_bootstrap.xlsx', engine='xlsxwriter')
    concatenated_df_acc.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()

    writer = pd.ExcelWriter(Path(data_dir) / 'precision_bootstrap.xlsx', engine='xlsxwriter')
    concatenated_df_precision.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
