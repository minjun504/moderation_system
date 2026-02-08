import torch
import numpy as np
from architecture import TierOneFilter, DistilBertRegressor
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

def csr_to_tensor(X):
    coo = X.tocoo()
    coords = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    vals = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(coords, vals, shape)

def train_one_epoch(model, loader, optimizer, loss_fxn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch.pop("target").unsqueeze(1)
        
        logits = model(**batch)
        loss = loss_fxn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, loss_fxn, device):
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch.pop("target").unsqueeze(1)
            logits = model(**batch)
            loss = loss_fxn(logits, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

def collate_dense(batch):
    xs = [item["x"] for item in batch]
    targets = [item["target"] for item in batch]
    
    if xs[0].is_sparse:
        xs = [x.to_dense() for x in xs]
    
    return {
        "x": torch.stack(xs),
        "target": torch.stack(targets)
    }

def get_embeddings(text_list, encoder_path, batch_size=32):
    model = ORTModelForFeatureExtraction.from_pretrained(encoder_path, file_name="model_quantized.onnx")
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    
    all_embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_vectors = outputs.last_hidden_state[:, 0, :].detach()
        all_embeddings.append(cls_vectors)
        
        del inputs, outputs, cls_vectors

    return torch.cat(all_embeddings, dim=0)
