from fastapi import FastAPI
from pydantic import BaseModel
from content_moderation_system.modeling.predict import load_tier_one, load_tier_two, get_embeddings
from content_moderation_system.features import csr_to_tensor
from content_moderation_system.config import MODELS_DIR
import torch

app = FastAPI(title="Toxic Comment Filter API")
device = torch.device("cpu")

vectorizer, tier1_model = load_tier_one(device)
tier2_model = load_tier_two(device)

class CommentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_toxicity(request: CommentRequest):
    text = request.text
    
    features = vectorizer.transform([text])
    feat_tensor = csr_to_tensor(features).to_dense().to(device)
    t1_logit = tier1_model(feat_tensor)
    prob = torch.sigmoid(t1_logit).item()
    
    model_used = "Tier 1 (Sieve)"
    
    if prob > 0.25:
        encoder_path = MODELS_DIR / "quantized_encoder"
        emb = get_embeddings([text], encoder_path).to(device)
        t2_logit = tier2_model(emb)
        prob = torch.sigmoid(t2_logit).item()
        model_used = "Tier 2 (BERT)"
        
    return {
        "toxicity_score": round(prob, 4),
        "is_toxic": prob > 0.5,
        "routed_logic": model_used
    }