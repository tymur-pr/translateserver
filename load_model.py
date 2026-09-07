from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import re
from pathlib import Path
import threading
import lang_dict

# Language data
_SRC_LANG = "en" # From which language
_DST_LANG = "de" # To which language

# Model data
_MODEL_NAME = str(Path(__file__).resolve().parent / "quantized_model")
_DEVICE = "cpu"

# Batching data
_BATCH_SIZE = 16

# Token data
_LENGTH_MULTIPLIER = {"hu": 3.0, "tr": 3.0}
_DEFAULT_LENGTH_MULTIPLIER = 2.0
_MIN_NEW_TOKENS = 10
_OVERFLOW_MULTIPLIER = 2.0
_MAX_RETRY_NEW_TOKENS = 512

# Regex to find dialogue markups like -> #$b# $h#$b#
_MARKUP_RE = re.compile(r"(?:\$[a-zA-Z0-9]|#|\+\+)+")

# Regex to find NPC data relationships like ->
# "NPC.Data.Kiarra": "Anton 'brother_Anton' Lorenzo 'oldest_brother_Lorenzo'",
_NPC_RELATION_RE = re.compile(r"(\S+) '([^']*)'")

# Load quantized model
print(f"Loading {_MODEL_NAME} onto {_DEVICE} ...")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, local_files_only=True)
_model = ORTModelForSeq2SeqLM.from_pretrained(_MODEL_NAME, provider="CPUExecutionProvider", local_files_only=True)

def translate_npc_data(value, translate_fn, src_lang=_SRC_LANG, tgt_lang=_DST_LANG):
    """Translate NPC_Data Name is never translated."""
    pairs = _NPC_RELATION_RE.findall(value)
    if not pairs:
        return value 

    # Underscores to spaces
    labels_to_translate = [label.replace("_", " ") for _, label in pairs if label.strip()]
    if labels_to_translate:
        translated_labels = translate_fn(labels_to_translate, src_lang=src_lang, tgt_lang=tgt_lang)
    else:
        translated_labels = []
    translated = iter(translated_labels)

    # Place _ in place
    parts = []
    for name, label in pairs:
        if not label.strip():
            parts.append(f"{name} ''")
        else:
            new_label = next(translated).strip()
            new_label = new_label.rstrip(".,!?;: ")
            new_label = new_label.replace(" ", "_")
            parts.append(f"{name} '{new_label}'")
    return " ".join(parts)

def _split_markup(text):
    """Split text into a list of (is_markup, chunk)"""
    pieces = []
    pos = 0
    for m in _MARKUP_RE.finditer(text):
        if m.start() > pos:
            pieces.append((False, text[pos:m.start()]))
        pieces.append((True, m.group()))
        pos = m.end()
    if pos < len(text):
        pieces.append((False, text[pos:]))
    return pieces

def _lang_choice(code):
    """Language from _LANG_CODE_MAP"""
    try:
        return lang_dict._LANG_CODE_MAP[code]
    except KeyError:
        raise ValueError(f"Unsuported code {code}, suported codes {lang_dict._LANG_CODE_MAP}")

_tokenizer_lock = threading.Lock()
def _batch_generation(batch_sentences, src_lang, tgt_lang):
    """Translate a single batch of sentences."""
    # Set language
    src_code = _lang_choice(src_lang)
    tgt_code = _lang_choice(tgt_lang)

    with _tokenizer_lock:
        _tokenizer.src_lang = src_code
        inputs = _tokenizer(batch_sentences, return_tensors="pt", padding=True).to(_DEVICE)
        forced_bos_token_id = _tokenizer.convert_tokens_to_ids(tgt_code)

    # Token length
    max_input_len = inputs["input_ids"].shape[1]
    multiplier = _LENGTH_MULTIPLIER.get(tgt_lang, _DEFAULT_LENGTH_MULTIPLIER)
    max_new_tokens = max(_MIN_NEW_TOKENS, int(max_input_len * multiplier))

    # Translation
    output_ids = _model.generate(**inputs,
                            forced_bos_token_id=forced_bos_token_id,
                            num_beams=2,
                            no_repeat_ngram_size=3,
                            max_new_tokens=max_new_tokens)
    translations = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Overflow
    eos_id = _tokenizer.eos_token_id
    for i, row in enumerate(output_ids):
        if eos_id is not None and eos_id in row.tolist():
            continue

        # Retry for big dialogues
        retry_cap = min(_MAX_RETRY_NEW_TOKENS, int(max_new_tokens * _OVERFLOW_MULTIPLIER))
        with _tokenizer_lock:
            _tokenizer.src_lang = src_code
            retry_inputs = _tokenizer([batch_sentences[i]], return_tensors="pt", padding=True).to(_DEVICE)
        retry_ids = _model.generate(**retry_inputs,
                                    forced_bos_token_id=forced_bos_token_id,
                                    num_beams=2,
                                    no_repeat_ngram_size=3,
                                    max_new_tokens=retry_cap)
        translations[i] = _tokenizer.batch_decode(retry_ids, skip_special_tokens=True)[0]
    return translations

@torch.no_grad()
def translate(sentences, src_lang=_SRC_LANG, tgt_lang=_DST_LANG):
    """Translate a list of sentences."""
    # Split every sentence
    split = [_split_markup(sentence) for sentence in sentences]

    # Flatten translatable text
    flat_sentences = []
    origin = []
    for s_idx, pieces in enumerate(split):
        for p_idx, (is_markup, chunk) in enumerate(pieces):
            if not is_markup and chunk.strip():
                flat_sentences.append(chunk)
                origin.append((s_idx, p_idx))


    # Cluster by legth
    indexed = sorted(enumerate(flat_sentences), key=lambda pair: len(pair[1]))

    # Translate batches
    sorded_translations = []
    for pos in range(0, len(indexed), _BATCH_SIZE):
        part = indexed[pos : pos + _BATCH_SIZE]
        part_word = [sentence for _, sentence in part]
        sorded_translations.extend(_batch_generation(part_word, src_lang, tgt_lang))

    # Revert to input order
    res = [None] * len(flat_sentences)
    for (index, _), translated in zip(indexed, sorded_translations):
        res[index] = translated

    # Put back markup
    for flat_idx, (s_idx, p_idx) in enumerate(origin):
        split[s_idx][p_idx] = (False, res[flat_idx])
    return ["".join(chunk for _, chunk in pieces) for pieces in split]

def _debug_translate(path, src_lang = _SRC_LANG, tgt_lang = _DST_LANG):
    """For translation debuging"""
    _NPC_KEY_RE = re.compile(r"^NPC[._]Data\.")
    import json5
    with open(path, "r", encoding="utf-8") as file:
        data = json5.load(file)

    npc_keys = [k for k in data if _NPC_KEY_RE.match(k)]
    plain_keys = [k for k in data if k not in npc_keys]

    plain_translations = translate([data[k] for k in plain_keys], src_lang=src_lang, tgt_lang=tgt_lang)
    npc_translations = [translate_npc_data(data[k], translate, src_lang=src_lang, tgt_lang=tgt_lang) for k in npc_keys]

    result = dict(zip(npc_keys, npc_translations))
    result.update(zip(plain_keys, plain_translations))

    print({k: result[k] for k in data})
    return result

if __name__ == "__main__":
    # Time benchmarking
    import time
    import json5
    start_time = time.perf_counter()

    output_lang = "de"
    result = _debug_translate("default.json", tgt_lang=output_lang)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Execution Time: {execution_time:.4f}")

    with open(f"{output_lang}.json", "w",encoding="utf-8") as file:
        json5.dump(result,file,indent=4, quote_keys=True,ensure_ascii=False)