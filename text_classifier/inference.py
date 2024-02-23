import pandas as pd
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from tqdm import tqdm, trange
from ast import literal_eval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("Number of GPU available:{} --> {} \n".format(n_gpu,torch.cuda.get_device_name()))

test_df = pd.read_csv('path_to_test_all_csv')
test_label_cols = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening', 'Tree in bud', 'Thymic remnant']
num_labels = len(test_label_cols)
test_labels = list(np.zeros((len(test_df),num_labels)))


# Gathering input data
test_comments = []

for data in test_df['Report Impression']:
    test_comments.append(str(data))
#test_comments = list(test_df['Report Impression'].values)

max_length = 512
batch_size= 32
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

# Encoding input data
test_encodings = tokenizer.batch_encode_plus(test_comments,max_length=max_length,pad_to_max_length=True)
test_input_ids = test_encodings['input_ids']
test_token_type_ids = test_encodings['token_type_ids']
test_attention_masks = test_encodings['attention_mask']

# Make tensors out of data
test_inputs = torch.tensor(test_input_ids)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_attention_masks)
test_token_types = torch.tensor(test_token_type_ids)
# Create test dataloader
test_data = TensorDataset(test_inputs, test_masks, test_labels, test_token_types)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
# Save test dataloader
#torch.save(test_dataloader,'test_data_loader')

# Test
device2 = torch.device('cpu')
model = BertForSequenceClassification.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", num_labels=num_labels)
model_path = "path_to_text_classifier_model_pth"
model.load_state_dict(torch.load(model_path, map_location=device2))
# Put model in evaluation mode to evaluate loss on the validation set
model.eval()
model.cuda()

#track variables
logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

print("test1")
# Predict
for i, batch in tqdm(enumerate(test_dataloader)):
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
#tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
pred_labels = [item for sublist in pred_labels for item in sublist]
true_labels = [item for sublist in true_labels for item in sublist]
# Converting flattened binary values to boolean values
#true_bools = [tl==1 for tl in true_labels]

pred_labels = np.array(pred_labels)

pred_labels[pred_labels>=0.5]=1
pred_labels[pred_labels<0.5]=0 #boolean output after thresholding

pred_labels = pred_labels.astype(np.uint8)

columns = ['AccessionNo','Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening', 'Tree in bud', 'Thymic remnant']

infer = pd.DataFrame()
infer[columns[0]] = test_df['AccessionNo']

for col,i in zip(columns[1:],range(num_labels)):
  infer[col] = pred_labels[:,i]

print("Total number of data: ",len(infer),' \n')
print('Count of 1 per label: \n', infer[columns[1:]].sum(), '\n')
with open('all_data_statistics.txt', 'w') as f:
    s = "Total number of data: " + str(len(infer)) + ' \n\n'
    f.write(s)
    s = 'Count of 1 per label: \n' + str(infer[columns[1:]].sum()) + '\n'
    f.write(s)


infer.to_csv('inferred.csv',index=False)
