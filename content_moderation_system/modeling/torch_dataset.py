import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class SieveData(Dataset):
    def __init__(self, vectors, targets):
        self.vectors = vectors
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {"x": self.vectors[idx], "target": self.targets[idx]}

class BertDataset(Dataset):
    def __init__(self, text, target, maxlen=128):
        self.text = text
        self.target = target
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert_base_uncased")
        self.max_len = maxlen
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        text = str(self.text[idx])
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": torch.tensor(self.targets[idx], dtype=torch.float)
        }
