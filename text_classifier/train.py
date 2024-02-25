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

save_path = 'Results'
save_path = get_unique_folder(save_path)
os.mkdir(save_path)

print('Results will be saved to ',save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ",device)
if device == 'cuda':
    n_gpu = torch.cuda.device_count()
    print("Number of GPU available:{} --> {} \n".format(n_gpu,torch.cuda.get_device_name()))


df = pd.read_csv('path_to_train_csv')
df2 = pd.read_csv('path_to_valid_csv')

print('average sentence length: ', df['Report Impression'].str.split().str.len().mean())
print('stdev sentence length: ', df['Report Impression'].str.split().str.len().std())


cols = df.columns
label_cols = list(cols[2:])
num_labels = len(label_cols)
print('Label columns: ', label_cols)

# Create dataloader

dataloaders = {}

max_length = 512
num_workers = 4

batch_size = 32
train_data = CTDataset(df, num_labels, label_cols, max_length)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
dataloaders['train'] = train_dataloader

test_data = CTDataset(df2, num_labels, label_cols, max_length)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)
dataloaders['val'] = test_dataloader


model = CTBertClassifier(n_classes=num_labels)
model = model.to(device)

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
epochs = 1000


trainer = ModelTrainer(model,
                       dataloaders,
                       num_labels,
                       epochs,
                       optimizer,
                       device,
                       save_path,
                       label_cols)
start = time.time()
print('----------------------Starting Training----------------------')
print("Number of epochs: ",epochs)
cm, clf_report = trainer.launch_training()

finish = time.time()
print('---------------------------------------------------------------')
print('Training_complete')
print('Training time: ',finish-start)
print(clf_report)

pickle.dump(clf_report, open('test_classification_report.txt','wb'))
pickle.dump(cm, open('test_confusion_matrix.txt','wb'))


