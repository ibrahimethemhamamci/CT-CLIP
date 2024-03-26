import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel

class RadBertClassifier(nn.Module):
    def __init__(self,n_classes=10):
      super().__init__()
    
      self.config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
      self.model = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=self.config)
    
      self.classifier=nn.Linear(self.model.config.hidden_size,n_classes) 
        
    def forward(self,input_ids, attn_mask):
        output = self.model(input_ids=input_ids,attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)
              
        return output
    

'''
tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m',do_lower_case=True)
model = RadBertClassifier(5)

print(model.eval())


text = ["There is a pleural effusion in the chest","There is a cardiomegaly in the chest"]
encoded_input = tokenizer(text, return_tensors='pt',max_length=512,padding='max_length')
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
out = model(input_ids,attention_mask)
'''
