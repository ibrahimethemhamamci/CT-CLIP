import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy
from transformers import AutoConfig, AutoTokenizer, AutoModel

class CTBertClassifier(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', config=self.config, trust_remote_code=True)
        self.pooler = nn.Sequential(
        nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
        nn.Tanh()
        )
        
        self.dropout = nn.Dropout(p=0.25)  # Separate dropout layer
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
            
    def load_pretrained(self,model_path):
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        m_weights = {k[len("model."):]: v for k, v in weights.items() if k.startswith("model.")}
        self.model.load_state_dict(m_weights)
        
    def forward(self,input_ids, attn_mask):
       # Extracting outputs from the model
        outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)

        pooled_output = self.pooler(outputs.last_hidden_state[:, 0])

        # Applying dropout to the pooled output
        pooled_output = self.dropout(pooled_output)

        return self.classifier(pooled_output)
    
