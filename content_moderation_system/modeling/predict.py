import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np

from content_moderation_system.config import PROCESSED_DATA_DIR, MODELS_DIR
from content_moderation_system.features import csr_to_tensor
from content_moderation_system.modeling.architecture import TierOneFilter
from content_moderation_system.modeling.utils import get_embeddings

class VectorRegressor(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.regressor = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x):
        return self.regressor(x)

def load_tier_one(device):
    vectorizer_path = MODELS_DIR / "tier1_vectorizer.pkl"
    model_path = MODELS_DIR / "tier1_model.pt"

    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError("Tier 1 artifacts not found. Run train.py first.")

    vectorizer = joblib.load(vectorizer_path)
    
    vocab_size = len(vectorizer.get_feature_names_out())
    model = TierOneFilter(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return vectorizer, model

def load_tier_two(device):
    model_path = MODELS_DIR / "tier2_head.pt"

    if not model_path.exists():
        raise FileNotFoundError("Tier 2 model not found. Run train.py first.")

    model = VectorRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def predict(df_test, device):
    vectorizer, t1_model = load_tier_one(device)
    
    texts = df_test["comment_text"].tolist()
    
    features = vectorizer.transform(texts)
    features_tensor = csr_to_tensor(features).to_dense().to(device)
    
    with torch.no_grad():
        t1_logits = t1_model(features_tensor)
        t1_probs = torch.sigmoid(t1_logits).cpu().numpy().flatten()

    uncertain_mask = t1_probs > 0.25
    uncertain_indices = np.where(uncertain_mask)[0]
    uncertain_texts = [texts[i] for i in uncertain_indices]
    
    final_probs = t1_probs.copy()
    
    if len(uncertain_texts) > 0:
        t2_model = load_tier_two(device)
        encoder_path = MODELS_DIR / "toxic_encoder_quantized"
        
        embeddings = get_embeddings(uncertain_texts, encoder_path).to(device)
        
        with torch.no_grad():
            t2_logits = t2_model(embeddings)
            t2_probs = torch.sigmoid(t2_logits).cpu().numpy().flatten()
        
        final_probs[uncertain_mask] = t2_probs

    return final_probs, uncertain_mask

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_path = PROCESSED_DATA_DIR / "processed_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")
        
    df_test = pd.read_csv(test_path).dropna(subset=["comment_text"])
    
    probs, routed_flag = predict(df_test, device)
    
    df_test["predicted_probability"] = probs
    df_test["routed_to_bert"] = routed_flag
    
    output_path = PROCESSED_DATA_DIR / "test_predictions.csv"
    df_test.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")