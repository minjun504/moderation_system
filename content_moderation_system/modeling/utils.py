import torch
import optuna
from architecture import TierOneFilter, DistilBertRegressor

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
    """
    Custom collator for SieveData.
    Input: List of dicts [{'x': sparse_tensor, 'target': tensor}, ...]
    Output: Dict {'x': dense_batch, 'target': target_batch}
    """
    xs = [item["x"] for item in batch]
    targets = [item["target"] for item in batch]
    
    if xs[0].is_sparse:
        xs = [x.to_dense() for x in xs]
    
    return {
        "x": torch.stack(xs),
        "target": torch.stack(targets)
    }