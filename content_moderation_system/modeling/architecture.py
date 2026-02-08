import torch
import torch.nn as nn
from transformers import DistilBertModel

class TierOneFilter(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.linear = nn.Linear(in_features=vocab_size, out_features=1)
        
    def forward(self, x):
        return self.linear(x)

class DistilBertRegressor(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        # self.distilbert = DistilBertModel.from_pretrained(model_name)
        # for params in self.distilbert.parameters():
        #     params.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(in_features=768, out_features=1)
    
    def forward(self, x):
        # bert_output = self.distilbert(input_ids=id, attention_mask=attention)
        # cls_vec = bert_output.last_hidden_state[:, 0, :]
        # x = self.dropout(cls_vec)
        score = self.regressor(self.dropout(x))
        return score