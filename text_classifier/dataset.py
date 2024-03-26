#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 01:18:28 2023

@author: furkan
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import sys
import os
import pandas as pd

from augmentation import TextAugment


class CTDataset(Dataset):
    
    def __init__(self, data_files, class_count, label_cols, max_length, augment = False ,infer=False):
        
        
        self.data_files = data_files
        self.class_count = class_count
        
        self.tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m',do_lower_case=True)
        self.max_length = 512
        self.label_cols = label_cols
        self.infer = infer
        self.augment = augment

        if self.augment:
            self.txt_augment = TextAugment()
        
        
    def __len__(self):
        return len(self.data_files)
    
    
    def __getitem__(self,idx):
        text_comment = self.data_files['report_text'][idx]
        text_comment = str(text_comment) if not isinstance(text_comment, str) else text_comment
        if pd.isna(text_comment):
            text_comment = " "

        if self.augment:
            text_comment = self.txt_augment.random_shuffle(text_comment)

        encodings = self.tokenizer(text_comment, return_tensors='pt',max_length=self.max_length,padding='max_length',truncation=True)
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask'] 
        
        if self.infer == False:
            label = self.data_files[self.label_cols].values[idx]
            label = torch.from_numpy(label).float()
            return input_ids, attention_mask, label
        
        else:            
            return input_ids, attention_mask           
        
