from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Model data
_MODEL_NAME = "google/madlad400-3b-mt"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Some models have different codes
_LANG_CODE_MAP = {
    "en": "en",
    "de": "de",
    "es": "es",
    "pt": "pt",
    "pt-BR": "pt",
    "ru": "ru",
    "ja": "ja",
    "zh": "zh",
    "zh-CN": "zh",
    "fr": "fr",
    "it": "it",
    "ko": "ko",
    "tr": "tr",
    "hu": "hu",
}

# Batching data
_BATCH_SIZE = 16

# Load model
print(f"Loading {_MODEL_NAME} onto {_DEVICE} ...")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME).to(_DEVICE)
_model.eval()

def _lang_choice(code):
    """Language from _LANG_CODE_MAP"""
    try:
        return _LANG_CODE_MAP[code]
    except KeyError:
        raise ValueError(f"Unsuported code {code}, suported codes {_LANG_CODE_MAP}")

def _batch_generation(batch_sentences, tgt_lang):
    """Translate a single batch of sentences."""
    # Set language
    tgt_code = _lang_choice(tgt_lang)
    tagged_sentences = [f"<2{tgt_code}> {sentence}" for sentence in batch_sentences]

    # Output as PyTorch + keep padding=True
    inputs = _tokenizer(tagged_sentences, return_tensors="pt", padding=True).to(_DEVICE)
    output_ids = _model.generate(**inputs, 
                                num_beams=1,
                                no_repeat_ngram_size=3)
    
    return _tokenizer.batch_decode(output_ids, skip_special_tokens=True)

@torch.no_grad()
def translate(sentences, src_lang="en", tgt_lang="fr"):
    """Translate a list of sentences."""
    # Cluster by legth
    indexed = sorted(enumerate(sentences), key=lambda pair: len(pair[1]))

    # Translate batches
    sorded_translations = []
    for pos in range(0, len(indexed), _BATCH_SIZE):
        part = indexed[pos : pos + _BATCH_SIZE]
        part_word = [sentence for _, sentence in part]
        sorded_translations.extend(_batch_generation(part_word, tgt_lang))

    # Revert to input order
    res = [None] * len(sentences)
    for (index, _), translated in zip(indexed, sorded_translations):
        res[index] = translated
    return res

def _debug_translate(path, lang = "ru"):
    """For translation debuging"""
    import json5
    with open(path, "r", encoding="utf-8") as file:
        data = json5.load(file)
    print(translate(list(data.values()),tgt_lang = lang))

if __name__ == "__main__":
    # Time benchmarking
    import time
    start_time = time.perf_counter()

    _debug_translate("default.json", lang="ru")

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Execution Time: {execution_time:.4f}")