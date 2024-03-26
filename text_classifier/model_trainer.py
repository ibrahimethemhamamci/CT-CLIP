#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 20:43:56 2023

@author: furkan
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
from torch.nn import BCEWithLogitsLoss
from scipy.special import expit
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
import time
import copy
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

class ModelTrainer:
    def __init__(self,
                 model,
                 dataloaders,
                 num_class,
                 epochs,
                 optimizer,
                 scheduler,
                 device,
                 save_path,
                 test_label_cols,
                 save_in=10,
                 early_stop=100,
                 threshold = 0.5):
        
        self.model = model
        self.dataloaders = dataloaders
        self.num_class = num_class
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.threshold = threshold
        self.early_stop = early_stop
        self.save_in = save_in
        self.test_label_cols = test_label_cols
        
        
    def launch_training(self):
        best_loss = 100
        
        best_model = copy.deepcopy(self.model.state_dict())
        no_improvement_for = 0
        
        for ep in range(self.epochs):
            #print("Learning_rate: ", self.scheduler.get_last_lr())
            train_loss, train_acc, train_f1_macro, train_f1_micro = self.train(ep)
            test_loss,pred_labels,test_labels = self.test(ep)
            if isinstance(self.scheduler, CosineAnnealingWarmupRestarts):
                print("Learning_rate: ", self.scheduler.get_lr())
                self.scheduler.step()

            elif isinstance(self.scheduler, ReduceLROnPlateau):
                print("Learning_rate: ", self.scheduler.optimizer.param_groups[0]['lr'])
                self.scheduler.step(test_loss)
            no_improvement_for += 1

            if(test_loss<best_loss):
                best_model = copy.deepcopy(self.model.state_dict())
                best_loss = test_loss
                no_improvement_for = 0
                print('Best Loss!!')
                torch.save(self.model.state_dict(),os.path.join(self.save_path,"RadBertClassifier_best.pth"))
            
            if(ep%self.save_in == 0):
                torch.save(self.model.state_dict(),os.path.join(self.save_path,f"RadBertClassifier_{ep}.pth"))
                
            if no_improvement_for == self.early_stop:
                break
        
        torch.save(self.model.state_dict(),os.path.join(self.save_path,f"RadBertClassifier_last.pth"))
        
        #Load the best model
        print("-------Loading the best model-------")
        self.model.load_state_dict(best_model)
        test_loss,pred_labels,test_labels = self.test(ep)
        cm = multilabel_confusion_matrix(test_labels,pred_labels)
        clf_report = classification_report(test_labels,pred_labels,target_names=self.test_label_cols)
        
        return cm, clf_report
    
    def train(self,epoch):
        self.model.train()
        train_loss = 0
        nb_steps = 0
        nb_samples = 0
        train_labels = np.zeros(self.num_class).reshape(1,self.num_class)
        pred_logits = np.zeros(self.num_class).reshape(1,self.num_class)
        time_pre_epoch = time.time()
        for input_ids,attention_mask,labels in tqdm(self.dataloaders['train']):
            input_ids = torch.squeeze(input_ids)
            attention_mask = torch.squeeze(attention_mask)
            self.optimizer.zero_grad()
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(input_ids,attention_mask)
            
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits,labels)
            
            train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()),axis=0)
            pred_logits = np.concatenate((pred_logits,logits.detach().cpu().numpy()),axis=0)
            train_loss += loss.item()
            loss.backward()

            self.optimizer.step()
            
            nb_steps += 1
            nb_samples += input_ids.shape[0]
        
        train_loss = train_loss/nb_steps
        train_labels = train_labels[1:]
        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        
        pred_labels[pred_labels>=self.threshold]=1
        pred_labels[pred_labels<self.threshold]=0
        
        f1_macro = f1_score(train_labels,pred_labels,average='macro')*100
        f1_micro = f1_score(train_labels,pred_labels,average='micro')*100
        accuracy = accuracy_score(train_labels.flatten(), pred_labels.flatten())*100
         
        print('Train - ep: {}/{}, loss: {:.3f}, acc: {:.3f}, f1_macro: {:.3f}, f1_micro: {:.3f}, {}s'.format(
           epoch + 1, self.epochs, train_loss,
           accuracy, f1_macro, f1_micro, int(time.time() - time_pre_epoch)))

        return train_loss, accuracy, f1_macro, f1_micro

    def test(self,epoch):
        self.model.eval()
        test_loss = 0
        nb_steps = 0
        nb_samples = 0
        test_labels = np.zeros(self.num_class).reshape(1,self.num_class)
        pred_logits = np.zeros(self.num_class).reshape(1,self.num_class)
        time_pre_epoch = time.time()
        
        with torch.no_grad():
            for input_ids,attention_mask,labels in tqdm(self.dataloaders['val']):
                input_ids = torch.squeeze(input_ids)
                attention_mask = torch.squeeze(attention_mask)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(input_ids,attention_mask)
                
                loss_func = BCEWithLogitsLoss()
                loss = loss_func(logits,labels)
                
                test_labels = np.concatenate((test_labels,labels.detach().cpu().numpy()),axis=0)
                pred_logits = np.concatenate((pred_logits,logits.detach().cpu().numpy()),axis=0)
                
                test_loss += loss.item()
                
                nb_steps += 1
                nb_samples += input_ids.shape[0]
        
        test_loss = test_loss/nb_steps
        test_labels = test_labels[1:]
        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        
        pred_labels[pred_labels>=self.threshold]=1
        pred_labels[pred_labels<self.threshold]=0
        
        f1_macro = f1_score(test_labels,pred_labels,average='macro')*100
        f1_micro = f1_score(test_labels,pred_labels,average='micro')*100
        accuracy = accuracy_score(test_labels.flatten(), pred_labels.flatten())*100
        

         
        print('Test - ep: {}/{}, loss: {:.3f}, acc: {:.3f}, f1_macro: {:.3f}, f1_micro: {:.3f}, {}s'.format(
           epoch + 1, self.epochs, test_loss,
           accuracy, f1_macro, f1_micro, int(time.time() - time_pre_epoch)))
        
        return test_loss, pred_labels, test_labels
    
    def infer(self):
        self.model.eval()
        pred_logits = np.zeros(self.num_class).reshape(1,self.num_class)

        with torch.no_grad():
            for input_ids,attention_mask in tqdm(self.dataloaders['test']):
                input_ids = torch.squeeze(input_ids)
                attention_mask = torch.squeeze(attention_mask)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                logits = self.model(input_ids,attention_mask)
                
                pred_logits = np.concatenate((pred_logits,logits.detach().cpu().numpy()),axis=0)
                
        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        
        pred_labels[pred_labels>=self.threshold]=1
        pred_labels[pred_labels<self.threshold]=0
        
        return pred_labels
    
    
    
    
    
