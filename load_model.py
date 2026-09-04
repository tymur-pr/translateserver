from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Model data
_MODEL_NAME = "facebook/nllb-200-distilled-600M"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print(f"Loading {_MODEL_NAME} onto {_DEVICE} ...")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME).to(_DEVICE)
_model.eval()

@torch.no_grad()
def translate(sentences, src_lang="eng_Latn", tgt_lang="fra_Latn"):
    """Translate a list of sentences."""
    _tokenizer.src_lang = src_lang

    # Output as PyTorch + keep padding=True
    inputs = _tokenizer(sentences, return_tensors="pt", padding=True).to(_DEVICE)
    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(tgt_lang)
    output_ids = _model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)

    return _tokenizer.batch_decode(output_ids, skip_special_tokens=True)