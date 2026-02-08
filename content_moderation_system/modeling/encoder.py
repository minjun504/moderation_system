from content_moderation_system.config import MODELS_DIR
from optimum.onnxruntime import ORTModelForFeatureExtraction
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

def setup_quantized_encoder():
    model_id = "unitary/toxic-bert"
    base_dir = MODELS_DIR/"onnx_base"
    final_dir = MODELS_DIR/"quantized_encoder"

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.config.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    model.save_pretrained(base_dir)
    quantize_dynamic(
        model_input=base_dir / "model.onnx",
        model_output=final_dir / "model_quantized.onnx",
        weight_type=QuantType.QUInt8
    )
    print("Done!")

if __name__ == "__main__":
    setup_quantized_encoder()