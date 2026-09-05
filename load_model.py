from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Model data
_MODEL_NAME = "facebook/nllb-200-distilled-600M"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_LANG_CODE_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "pt-BR": "por_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "zh-CN": "zho_Hans",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ko": "kor_Hang",
    "tr": "tur_Latn",
    "hu": "hun_Latn",
}

# Load model
print(f"Loading {_MODEL_NAME} onto {_DEVICE} ...")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME).to(_DEVICE)
_model.eval()

def _lang_choice(code):
    try:
        return _LANG_CODE_MAP[code]
    except KeyError:
        raise ValueError(f"Unsuported code {code}, suported codes {_LANG_CODE_MAP}")


@torch.no_grad()
def translate(sentences, src_lang="en", tgt_lang="fr"):
    """Translate a list of sentences."""
    _tokenizer.src_lang = src_lang

    # Output as PyTorch + keep padding=True
    inputs = _tokenizer(sentences, return_tensors="pt", padding=True).to(_DEVICE)
    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(_lang_choice(tgt_lang))
    output_ids = _model.generate(**inputs, 
                                 forced_bos_token_id=forced_bos_token_id,
                                 num_beams=4,
                                 no_repeat_ngram_size=3)
    return _tokenizer.batch_decode(output_ids, skip_special_tokens=True)

print(translate(["A gentle, sweet breakfast with the warm aroma of butter and maple."],tgt_lang="ru"))