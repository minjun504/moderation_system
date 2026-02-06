import torch
from content_moderation_system.modeling.architecture import DistilBertRegressor
from content_moderation_system.config import MODELS_DIR
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight

def export_quantized_model():
    model = DistilBertRegressor(model_name="martin-ha/distilbert-base-uncased-toxic-comments")

    model.eval()
    quantize_(model, int8_dynamic_activation_int8_weight())

    save_path = MODELS_DIR / "BERT_quantized_8bit.pt"
    torch.save(model.state_dict(), save_path)
    print("Done!")
