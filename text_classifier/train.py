import pandas as pd
import numpy as np
#import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from tqdm import tqdm, trange
from ast import literal_eval



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("Number of GPU available:{} --> {} \n".format(n_gpu,torch.cuda.get_device_name()))


df = pd.read_csv('path_to_train_text_classifier_csv')
df2 = pd.read_csv('path_to_valid_text_classifier_csv')

print('average sentence length: ', df['Report Impression'].str.split().str.len().mean())
print('stdev sentence length: ', df['Report Impression'].str.split().str.len().std())


cols = df.columns
label_cols = list(cols[2:])
num_labels = len(label_cols)
print('Label columns: ', label_cols)

df['one_hot_labels'] = list(df[label_cols].values)
df2['one_hot_labels'] = list(df2[label_cols].values)

train_labels = list(df.one_hot_labels.values)
test_labels = list(df2.one_hot_labels.values)
train_comments = list(df['Report Impression'].values)
test_comments = list(df2['Report Impression'].values)

# Change the max_length to 1024
max_length = 512
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)


train_encodings = tokenizer.batch_encode_plus(train_comments,max_length=max_length,pad_to_max_length=False)
test_encodings = tokenizer.batch_encode_plus(test_comments,max_length=max_length,pad_to_max_length=False)
tr_input_ids = train_encodings['input_ids']
test_input_ids = test_encodings['input_ids']
train_length = [len(i) for i in tr_input_ids]
test_length = [len(i) for i in test_input_ids]
print("Max tokenized text length in train data: ",max(train_length))
print("Max tokenized text length in test data: ",max(test_length))


train_encodings = tokenizer.batch_encode_plus(train_comments,max_length=max_length,pad_to_max_length=True) # tokenizer's encoding method
test_encodings = tokenizer.batch_encode_plus(test_comments,max_length=max_length,pad_to_max_length=True) # tokenizer's encoding method

#print('tokenizer outputs: ', train_encodings.keys())

tr_input_ids = train_encodings['input_ids'] # tokenized and encoded sentences
tr_token_type_ids = train_encodings['token_type_ids'] # token type ids
tr_attention_masks = train_encodings['attention_mask'] # attention masks

test_input_ids = test_encodings['input_ids'] # tokenized and encoded sentences
test_token_type_ids = test_encodings['token_type_ids'] # token type ids
test_attention_masks = test_encodings['attention_mask'] # attention masks

train_inputs = torch.tensor(tr_input_ids)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(tr_attention_masks)
train_token_types = torch.tensor(tr_token_type_ids)

validation_inputs = torch.tensor(test_input_ids)
validation_labels = torch.tensor(test_labels)
validation_masks = torch.tensor(test_attention_masks)
validation_token_types = torch.tensor(test_token_type_ids)

# Create dataloader

batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#torch.save(validation_dataloader,'validation_data_loader')
#torch.save(train_dataloader,'train_data_loader')


model = BertForSequenceClassification.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", num_labels=num_labels)
model.cuda()

# setting custom optimization parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5,correct_bias=True)

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs
epochs = 300
save_in = 10


for _ in trange(epochs, desc="Epoch"):

  # Training

  model.train()

  # Tracking variables
  tr_loss = 0 #running loss
  nb_tr_examples, nb_tr_steps = 0, 0

  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels, b_token_types = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()

    # Forward pass for multilabel classification
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    loss_func = BCEWithLogitsLoss()
    loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
    # loss_func = BCELoss()
    # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
    train_loss_set.append(loss.item())

    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    # scheduler.step()
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))

###############################################################################

  # Validation

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Variables to gather full output
  logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

  # Predict
  for i, batch in enumerate(validation_dataloader):
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels, b_token_types = batch
    with torch.no_grad():
      # Forward pass
      outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      b_logit_pred = outs[0]
      pred_label = torch.sigmoid(b_logit_pred)

      b_logit_pred = b_logit_pred.detach().cpu().numpy()
      pred_label = pred_label.to('cpu').numpy()
      b_labels = b_labels.to('cpu').numpy()

    tokenized_texts.append(b_input_ids)
    logit_preds.append(b_logit_pred)
    true_labels.append(b_labels)
    pred_labels.append(pred_label)

  # Flatten outputs
  pred_labels = [item for sublist in pred_labels for item in sublist]
  #print("pred_labels: ",pred_labels)
  true_labels = [item for sublist in true_labels for item in sublist]
  #print("true_labels: ",true_labels)

  # Calculate Accuracy
  threshold = 0.50
  pred_labels = np.array(pred_labels)
  true_labels = np.array(true_labels)
  pred_labels[pred_labels>=threshold]=1
  pred_labels[pred_labels<threshold]=0

  val_f1_accuracy = f1_score(true_labels,pred_labels,average='micro')*100

  val_flat_accuracy = accuracy_score(true_labels.flatten(), pred_labels.flatten())*100

  print('F1 Validation Accuracy: ', val_f1_accuracy)
  print('Flat Validation Accuracy: ', val_flat_accuracy)

  if _%save_in == 0:
    save_name = 'epoch' + str(_) + '_bert_model_ct.pth'
    torch.save(model.state_dict(), save_name)
    
  torch.save(model.state_dict(), 'bert_model_ct_last.pth')


