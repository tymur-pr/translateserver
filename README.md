# startranslate [!!! Be warned before you run this !!!](#be-warned-before-you-run-this)

A self-hosted machine translation server built for game localization workflows
that store dialogue and item text as flat JSON key/value pairs - the format
used by **Stardew Valley** mods (Content Patcher-style dialogue and item
data), and by many other games that follow the same "one JSON file per
language, one key per line of text" convention. Point it at a JSON file of
English source strings and it hands back the same structure translated into
the target language, while leaving game-specific markup and proper nouns
alone.

## What it does

- Loads a translation model **once, into RAM, at process startup** (not per
  request), so a running server only pays for inference per call, never for
  reloading the model.
- Exposes a small Flask API (`app.py`, served by Gunicorn) protected by an
  API key, so multiple internal tools/clients can share one running model
  instead of each needing their own copy in memory.
- Preserves structure that should never be touched by translation:
  - **Dialogue markup** - tokens like `#`, `#$b#`, `#$e#`, `$h`, `++`, and
    similar Stardew-style codes are stripped out before translation and
    stitched back in afterward untouched (`_split_markup` /  `_MARKUP_RE` in
    `load_model.py`).
  - **NPC relationship data** - values shaped like
    `"Anton 'brother_Anton' Lorenzo 'oldest_brother_Lorenzo'"` (seen in
    `NPC_Data.*` / `NPC.Data.*` keys) get parsed so that names are never sent
    through the model at all, and only the quoted relationship label is
    translated (`translate_npc_data` in `load_model.py`).
- Batches sentences together for throughput and automatically retries a
  sentence with a larger token budget if generation runs out of room before
  reaching a natural end.


- **`app.py`** - the Flask app: `POST /translate` (API-key protected) and
  `GET /health` (unauthenticated liveness check).
- **`helpers.py`** - the `@reqire_key` decorator checking the `x-api-key`
  header against `API_KEY` from `.env`.
- **`load_model.py`** - loads the tokenizer + model at import time and
  exposes `translate(sentences, src_lang, tgt_lang)`. This is the only file
  that talks to the ML stack, and it's also everything you need to translate
  text without running a server at all (see below).
- **`lang_dict.py`** - the `_LANG_CODE_MAP` dictionary mapping short language
  codes (`"ru"`, `"ja"`, ...) to the model's own internal language codes
  (FLORES-200 codes like `"rus_Cyrl"`, `"jpn_Jpan"`). This is the file to
  edit to add or remove supported languages.
- **`quantize_model.py`** - a one-off, offline script (never run by the
  server itself). It downloads the base model from Hugging Face, exports it
  to ONNX, and int8-quantizes it into `quantized_model/`. Run it once
  whenever you need to (re)build that folder; the Dockerfile just copies the
  already-built output in.

## Models used

This project has gone through two models in its committed history (see
`git log -- quantize_model.py` for the exact commits):

### `google/madlad400-3b-mt` (original)
- **Pros:** covers roughly 400 languages from a single checkpoint, strong
  general translation quality including many lower-resource languages,
  simple language selection (a `<2xx>` text tag prepended to the input).
- **Cons:** 3 billion parameters is heavy for CPU-only serving - the
  original benchmark translated ~100 lines in around 20 seconds. The
  quantized ONNX export came out to ~4.7GB on disk, partly because the
  non-merged decoder export duplicates decoder weights across the
  "with past" and "without past" graphs. RAM usage was high enough that
  running more than one Gunicorn worker wasn't realistic.

### `facebook/nllb-200-distilled-1.3B` (current)
- **Pros:** a distilled model - trained to compress a much larger NLLB-200
  teacher model down to 1.3B parameters while keeping most of its quality -
  so it's roughly 2.3x smaller than MADLAD-400-3B and noticeably faster on
  CPU, while still covering 200 languages. Exports and quantizes cleanly
  through the same `optimum`/ONNX Runtime pipeline already in place.
- **Cons:** still a substantial model to run without a GPU. Every line is
  translated in isolation - there's no awareness of the previous line,
  speaker, or scene, so tone/phrasing can be inconsistent across what's
  obviously one conversation. Proper nouns aren't protected by the model
  itself. Greedy decoding on long or unusually punctuated sentences can 
  occasionally skip a clause in the middle of the output - see the 
  `num_beams` note below.

### Other models evaluated, not adopted
- **`haoranxu/ALMA-7B`** - a LLaMA-2-based model fine-tuned specifically for
  translation, tried to see if an instruction-following LLM would handle
  names/context better than a classic encoder-decoder model. Rejected: it
  only supports 6 languages (en, de, cs, is, zh, ru) - nowhere near this
  project's language list - and is also heavier (7B params) with no speed
  benefit.
- **`utter-project/EuroLLM-1.7B-Instruct`** - a smaller, translation-tuned
  LLM that does cover this project's full language list. Prototyped but not
  carried forward into the committed pipeline.
- **`google-t5/t5-large`** - tested as a possible hosted-API option; only
  translates English→German/French/Romanian, far too narrow for this
  project's needs.
- **Hosted inference APIs** (Hugging Face Inference Providers) - explored as
  an alternative to self-hosting entirely. `facebook/nllb-200-distilled-1.3B`
  turned out not to be deployed on any Inference Provider at the time of
  testing, making this route unreliable for the model this project actually
  needs.

## Using it without a server (no Flask, just Python)

Everything the server does is really just `load_model.py` - the Flask layer
in `app.py` is only there to expose it over HTTP to other machines/processes.
If you have Python and the same environment set up (see `requirements.txt`),
you can translate text directly from a Python shell or script, no server
needed:

```bash
pip install -r requirements.txt
```

**One-time setup** - build the quantized model (downloads the base model
from Hugging Face and exports/quantizes it, ~1.3B params worth of weights):

```bash
python quantize_model.py
```

This creates a `quantized_model/` folder next to the scripts. You only need
to do this once (or again if you change `MODEL_NAME` in `quantize_model.py`).

**Then, translate directly:**

```python
import load_model

lines = ["Hello there!", "Where is the general store?"]
translated = load_model.translate(lines, src_lang="en", tgt_lang="de")
print(translated)
```

`import load_model` triggers the model load immediately (you'll see
`Loading .../quantized_model onto cpu ...` printed) - that happens once per
Python process, exactly like it does once per Gunicorn worker in the server.
After that, call `load_model.translate(...)` as many times as you like in
that same process.

For NPC relationship data specifically (`NPC_Data.*` / `NPC.Data.*` style
values), use `load_model.translate_npc_data(value, load_model.translate,
tgt_lang="de")` instead - see `_debug_translate` in `load_model.py` for a
complete worked example that reads a JSON file, routes each key through the
right function, and writes a translated JSON file back out.

# Be warned before you run this

- **This will load significant CPU load.** There's no GPU involved anywhere
  in this pipeline - every translation call runs the full model forward pass
  on your CPU. Expect your CPU usage to spike to (or near) 100% on however
  many cores ONNX Runtime is allowed to use, for as long as translation is
  running.
- **Translation is slow, and that's largely expected, not a bug.** Even
  quantized and running through ONNX Runtime, a 1.3B-parameter model doing
  autoregressive generation on CPU, with no GPU acceleration, is
  fundamentally a heavy workload. Translating dozens to hundreds of lines
  can take anywhere from several seconds to well over a minute depending on
  line length, batch size, and your CPU.
- **There isn't much more headroom to optimize this, at this model and on
  this hardware.** This project already applies essentially every low-risk
  optimization available: int8 quantization, ONNX Runtime instead of raw
  PyTorch, batching same-length sentences together, and a model already
  chosen specifically for being smaller than the original (MADLAD-400-3B).
  Meaningfully faster than this, on CPU, with comparable language coverage
  and quality, realistically requires one of: a genuinely smaller/different
  model (quality tradeoff), an actual GPU (hardware/deployment change), or a
  different inference engine entirely (like CTranslate2) - not a parameter
  tweak. Don't expect to tune your way to dramatically better performance
  without changing one of those three things.

## Parameters you can tune

All of these live inside `_batch_generation` in `load_model.py`:


- **`num_beams`** - beam search width. `2` (the current main-pass setting)
  is greedy decoding: fastest, but on long or unusually punctuated
  sentences it can occasionally skip a clause in the middle of the
  translation, since greedy search commits to one path with no ability to
  reconsider. Raising it (the retry path `4`) makes the model
  explore multiple candidate translations at once, meaningfully reducing -
  though not eliminating - that failure mode, at a roughly linear cost in
  compute (num_beams=4 is roughly 4x the decoding work of num_beams=1).
- **`no_repeat_ngram_size`** - currently `3`. Blocks the model from
  generating the same 3-token sequence twice in one output, which prevents
  degenerate repetition loops. Setting it lower (`1` or `2`) makes repeat
  blocking more aggressive, which can backfire on text that's *supposed* to
  repeat a word or short phrase naturally.
- **The token cap (`max_new_tokens`)** - not a single constant, but computed
  per call from a few settings:
  - `_MIN_NEW_TOKENS` (currently `10`) - the absolute floor, regardless of
    how short the input is.
  - `_DEFAULT_LENGTH_MULTIPLIER` (currently `2.0`) and the per-language
    overrides in `_LENGTH_MULTIPLIER` (currently `{"hu": 3.0, "tr": 3.0}`) -
    the token cap is `max(_MIN_NEW_TOKENS, input_token_count * multiplier)`.
    Languages that tend to need more tokens per idea than English (like
    Hungarian and Turkish) get a higher multiplier so they're less likely to
    get cut off mid-sentence.
  - `_MAX_RETRY_NEW_TOKENS` (currently `512`) and `_OVERFLOW_MULTIPLIER`
    (currently `2.0`) - control the retry path's larger budget when the
    first attempt didn't reach a natural end.
  - Raising these lets longer sentences finish without truncation, at the
    cost of a higher worst-case token budget (and therefore time) per call;
    lowering them tightens that ceiling but risks cutting off long lines.
- **`_BATCH_SIZE`** (currently `16`, top of `load_model.py`) - how many
  sentences get grouped into one padded batch. Higher values improve
  throughput (less per-call overhead) but increase peak RAM per batch;
  lower values do the opposite.
- **`_SRC_LANG` / `_DST_LANG`** (top of `load_model.py`, currently `"en"` and
  `"de"`) - the default source and target languages used whenever
  `translate()` or `translate_npc_data()` are called without explicitly
  passing `src_lang`/`tgt_lang` - e.g. calling `load_model.translate([...])`
  directly from a script, or running the `__main__` benchmark block.
  Changing either only changes what happens when a caller *omits* that
  argument - it doesn't restrict what can be requested; any code present in
  `lang_dict._LANG_CODE_MAP` can still be passed explicitly regardless of
  what these defaults are set to. This only governs the standalone/no-server
  usage path: the Flask API in `app.py` has its own separate defaults -
  `src_lang` falls back to `"en"` there too, but `tgt_lang` has no default at
  all and must be passed on every request - so changing `_SRC_LANG`/
  `_DST_LANG` here has no effect on what the server does.

## Changing / adding languages

Edit `_LANG_CODE_MAP` in **`lang_dict.py`** - it's a plain dictionary mapping
a short code (what you pass as `src_lang`/`tgt_lang`) to the model's FLORES-200
language code:

```python
_LANG_CODE_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    ...
}
```

To add a language, find its FLORES-200 code (NLLB-200's documentation lists
all 200) and add a `"xx": "xxx_Yyyy"` entry. To translate into/from it, just
pass that short code as `src_lang`/`tgt_lang` to `load_model.translate(...)` -
no other code changes needed. Passing a code that isn't in the dictionary
raises a clear `ValueError` naming the unsupported code, rather than failing
silently.
