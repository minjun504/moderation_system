import pandas as pd
import torch
import torch.nn as nn
import json
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

from content_moderation_system.config import PROCESSED_DATA_DIR, CONFIG_DIR, MODELS_DIR
from content_moderation_system.features import csr_to_tensor
from content_moderation_system.modeling.architecture import TierOneFilter, DistilBertRegressor
from content_moderation_system.modeling.torch_dataset import VectorDataset
from content_moderation_system.modeling.utils import train_one_epoch, get_embeddings, collate_dense

def load_best_params():
    param_path = CONFIG_DIR / "best_hyperparams.json"
    if not param_path.exists():
        raise FileNotFoundError("Run tune.py first to generate best_hyperparams.json")
    
    with open(param_path, "r") as f:
        return json.load(f)

def train_tier_one(df_train, params, device):
    X_text = df_train["comment_text"].values
    y_train = torch.tensor(df_train["target"].values).float().unsqueeze(1).to(device)

    ngram = tuple(map(int, params["ngram_range"].split(","))) if isinstance(params["ngram_range"], str) else params["ngram_range"]
    vectorizer = TfidfVectorizer(
        ngram_range=ngram, 
        max_features=params["max_features"]
    )
    X_vec = vectorizer.fit_transform(X_text)
    
    joblib.dump(vectorizer, MODELS_DIR / "tier1_vectorizer.pkl")

    vocab_size = len(vectorizer.get_feature_names_out())
    model = TierOneFilter(vocab_size=vocab_size).to(device)
    
    X_tensor = csr_to_tensor(X_vec)
    dataset = VectorDataset(X_tensor, y_train.cpu()) 
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_dense)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(5): 
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "tier1_model.pt")

def train_tier_two(df_train, params, device):    
    texts = df_train["comment_text"].tolist()
    y_train = torch.tensor(df_train["target"].values).float().unsqueeze(1).to(device)
    encoder_path = MODELS_DIR / "toxic_encoder_quantized"

    embeddings = get_embeddings(texts, encoder_path, batch_size=32).to(device)

    model = DistilBertRegressor(dropout_rate=params["dropout_rate"]).to(device)

    dataset = VectorDataset(embeddings.cpu(), y_train.cpu())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(5): 
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "tier2_head.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(PROCESSED_DATA_DIR / "processed_train.csv")
    
    best_params = load_best_params()

    if "sieve" in best_params:
        train_tier_one(df_train, best_params["sieve"], device)
    
    if "BERT" in best_params:
        train_tier_two(df_train, best_params["BERT"], device)
        
    print("\nAll models trained and saved successfully!")