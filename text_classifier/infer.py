# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 03:03:25 2024

@author: zehra
"""

import pandas as pd
import numpy as np
import os
import torch
import pickle
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.optim import AdamW
import time

from model_trainer import ModelTrainer
from classifier import CTBertClassifier
from dataset import CTDataset

def get_unique_folder(base_folder):
    counter = 1
    new_folder = base_folder
    
    while os.path.exists(new_folder):
        new_folder = f"{base_folder}{counter}"
        counter += 1
    
    return new_folder

save_path = 'Results_infer'
save_path = get_unique_folder(save_path)
os.mkdir(save_path)

print('Results will be saved to ',save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ",device)
if device == 'cuda':
    n_gpu = torch.cuda.device_count()
    print("Number of GPU available:{} --> {} \n".format(n_gpu,torch.cuda.get_device_name()))


df = pd.read_csv('path_to_test_all_csv')

label_cols = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening', 'Tree in bud', 'Thymic remnant']
num_labels = len(label_cols)
print('Label columns: ', label_cols)

# Create dataloader

dataloaders = {}

max_length = 512
num_workers = 4

batch_size = 32

test_data = CTDataset(df, num_labels, label_cols, max_length, infer=True)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)
dataloaders['test'] = test_dataloader


model_path = 'path_to_CTBertClassifier_best_pth'

model = CTBertClassifier(n_classes=num_labels)
model.load_pretrained(model_path)
model = model.to(device)
print(model.eval())

#No need to set up for inference
# setting custom optimization parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
# optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization
epochs = 0


trainer = ModelTrainer(model,
                       dataloaders,
                       num_labels,
                       epochs,
                       optimizer,
                       device,
                       save_path,
                       label_cols)
start = time.time()
print('----------------------Starting Inferring----------------------')
predicted_labels = trainer.infer()

finish = time.time()
print('---------------------------------------------------------------')
print('Inferring Complete')
print('Infer time: ',finish-start)

columns = ['AccessionNo','Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening', 'Tree in bud', 'Thymic remnant']

inferred_data = pd.DataFrame()
inferred_data[columns[0]] = df['AccessionNo']

for col,i in zip(columns[1:],range(num_labels)):
    inferred_data[col] = predicted_labels[:,i]

save_df = os.path.join(save_path,'inferred.csv')
inferred_data.to_csv(save_df,index=False)
print('Inferred data saved to: ',save_df)


