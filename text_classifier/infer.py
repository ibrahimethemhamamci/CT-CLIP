# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 03:03:25 2024

@author: zehra
"""

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.optim import AdamW
import time
import argparse as ap

from model_trainer import ModelTrainer
from classifier import RadBertClassifier
from dataset import CTDataset

parser = ap.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    help='model_path',
    required=True,
    default=None) 

parser.add_argument(
    '--dataset',
    help='multiple data file inference',
    required=False,
    default=None)

parser.add_argument(
    '--single_data',
    help='single data file inference',
    required=False,
    default=None)

parser.add_argument(
    '--save_path',
    help='single data file inference',
    required=False,
    default="./output/")
args = parser.parse_args()

def get_unique_folder(base_folder):
    counter = 1
    new_folder = base_folder
    
    while os.path.exists(new_folder):
        new_folder = f"{base_folder}{counter}"
        counter += 1
    
    return new_folder

if not args.dataset and not args.single_data:
  raise ValueError("Either --dataset or --single_data argument is required")

if args.dataset and args.single_data:
  raise ValueError("Both --dataset and --single_data arguments cannot be used simultaneously")

save_path = args.save_path
os.mkdir(save_path)

print('Results will be saved to ',save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ",device)
if device == 'cuda':
    n_gpu = torch.cuda.device_count()
    print("Number of GPU available:{} --> {} \n".format(n_gpu,torch.cuda.get_device_name()))

if args.dataset:
  df = pd.read_csv(os.path.join(args.dataset,'test.csv')) 
else:
  df = pd.read_csv(args.single_data) 

label_cols = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
num_labels = len(label_cols)
print('Label columns: ', label_cols)
print('\nNumber of test data: ',len(df))

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


model_path = args.checkpoint

model = RadBertClassifier(n_classes=num_labels)
model.load_state_dict(torch.load(model_path))
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
scheduler = None
# optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization
epochs = 0


trainer = ModelTrainer(model,
                       dataloaders,
                       num_labels,
                       epochs,
                       optimizer,
                       scheduler,
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

columns = ['AccessionNo','Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']

inferred_data = pd.DataFrame()
inferred_data[columns[0]] = df['AccessionNo']
inferred_data['report_text'] = df['report_text']

for col,i in zip(columns[1:],range(num_labels)):
    inferred_data[col] = predicted_labels[:,i]

save_df = os.path.join(save_path,'inferred.csv')
inferred_data.to_csv(save_df,index=False)
print('Inferred data saved to: ',save_df)


