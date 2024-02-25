#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 01:18:28 2023

@author: furkan
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image

import sys
import os
import pandas as pd



class CTDataset(Dataset):
    
    def __init__(self, data_files, class_count, label_cols, max_length, infer=False):
        
        
        self.data_files = data_files
        self.class_count = class_count
        
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True, trust_remote_code=True)
        self.max_length = 512
        self.label_cols = label_cols
        self.infer = infer
        
        
    def __len__(self):
        return len(self.data_files)
    
    
    def __getitem__(self,idx):
        text_comment = self.data_files['Report Impression'][idx]
        text_comment = str(text_comment) if not isinstance(text_comment, str) else text_comment
        if pd.isna(text_comment):
            text_comment = " "
        
        encodings = self.tokenizer(text_comment, return_tensors='pt',max_length=self.max_length,padding='max_length',truncation=True)
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask'] 
        
        if self.infer == False:
            label = self.data_files[self.label_cols].values[idx]
            label = torch.from_numpy(label).float()
            return input_ids, attention_mask, label
        
        else:            
            return input_ids, attention_mask           
        
