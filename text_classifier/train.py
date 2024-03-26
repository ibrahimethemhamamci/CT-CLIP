import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.optim import AdamW
import time
import argparse as ap

from model_trainer import ModelTrainer
from classifier import RadBertClassifier
from dataset import CTDataset
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = ap.ArgumentParser()
parser.add_argument(
    '--dataset',
    help='dataset',
    required=True,
    default=None) 

parser.add_argument(
    '--augment',
    help='augmentation',
    required=False,
    default=0,
    type=int) 

parser.add_argument(
    '--scheduler',
    help='scheduler, cawr or rlop',
    required=False,
    default=None) 

args = parser.parse_args()

augment = True if args.augment == 1 else False

def get_unique_folder(base_folder):
    counter = 1
    new_folder = base_folder
    
    while os.path.exists(new_folder):
        new_folder = f"{base_folder}{counter}"
        counter += 1
    
    return new_folder

save_path = 'Results_'+os.path.basename(args.dataset)
save_path = get_unique_folder(save_path)
os.mkdir(save_path)

print('Results will be saved to ',save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ",device)
if device == 'cuda':
    n_gpu = torch.cuda.device_count()
    print("Number of GPU available:{} --> {} \n".format(n_gpu,torch.cuda.get_device_name()))


df = pd.read_csv(os.path.join(args.dataset,'train.csv')) 
df2 = pd.read_csv(os.path.join(args.dataset,'val.csv'))

print('average sentence length: ', df['report_text'].str.split().str.len().mean())
print('stdev sentence length: ', df['report_text'].str.split().str.len().std())


cols = df.columns
label_cols = list(cols[2:])
num_labels = len(label_cols)
print('Label columns: ', label_cols)

# Create dataloader

dataloaders = {}

max_length = 512
num_workers = 4

batch_size = 32
train_data = CTDataset(df, num_labels, label_cols, max_length, augment = args.augment)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
dataloaders['train'] = train_dataloader

val_data = CTDataset(df2, num_labels, label_cols, max_length)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)
dataloaders['val'] = val_dataloader


model = RadBertClassifier(n_classes=num_labels)
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

lr_rate = 2e-5
optimizer = AdamW(optimizer_grouped_parameters,lr=lr_rate)
# optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization
epochs = 1000
w_steps = 50
cycle_step = 200
if args.scheduler == 'cawr':
  scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                            first_cycle_steps=cycle_step,
                                            cycle_mult = 1,
                                            max_lr = lr_rate,
                                            min_lr = 2e-7,
                                            warmup_steps = w_steps)
elif args.scheduler =='rlop':
  scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr = 2e-6, factor=0.1, patience=25)
else:
  scheduler = None

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
print('----------------------Starting Training----------------------')
print("Number of epochs: ",epochs)
cm, clf_report = trainer.launch_training()

finish = time.time()
print('---------------------------------------------------------------')
print('Training_complete')
print('Training time: ',finish-start)
print(clf_report)

with open(os.path.join(save_path, 'test_classification_report.txt'), 'w') as file:
  file.write(clf_report)

# Save confusion matrix
np.save(os.path.join(save_path, 'test_confusion_matrix.npy'), cm)


