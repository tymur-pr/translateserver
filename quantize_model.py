import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_NAME = "google/madlad400-3b-mt"
EXPORT_DIR = Path("./onnx_model")
QUANTIZED_DIR = Path("./quantized_model")

# Model to ONNX
onnx_model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAME, export=True)
onnx_model.save_pretrained(EXPORT_DIR)
AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(EXPORT_DIR)

# Quantize each component separately
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
# If your CPU doesn't support AVX-512 VNNI, use AutoQuantizationConfig.avx2() instead.

component_files = [f.name for f in EXPORT_DIR.glob("*.onnx")]
QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

for file_name in component_files:
    print(f"Quantizing {file_name} ...")
    quantizer = ORTQuantizer.from_pretrained(EXPORT_DIR, file_name=file_name)
    quantizer.quantize(save_dir=QUANTIZED_DIR, 
                       quantization_config=dqconfig,
                       use_external_data_format=True)

    # ORTQuantizer rename
    quantized_name = QUANTIZED_DIR / file_name.replace(".onnx", "_quantized.onnx")
    plain_name = QUANTIZED_DIR / file_name
    if quantized_name.exists():
        quantized_name.replace(plain_name)

# Copy tokenizer/config files + quantized ONNX
for pattern in ("*.json", "*spiece*"):
    for extra_file in EXPORT_DIR.glob(pattern):
        shutil.copy(extra_file, QUANTIZED_DIR / extra_file.name)

print(f"Done. Quantized model + tokenizer are in {QUANTIZED_DIR.resolve()}")