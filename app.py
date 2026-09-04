from flask import Flask, jsonify, request
import load_model, helpers

app = Flask(__name__)
app.json.ensure_ascii = False

# Main route to translate
@app.route("/translate", methods=["POST"])
@helpers.reqire_key
def translate_endpoint():
    sentences = request.get_json(silent=True)

    # Check request
    if not isinstance(sentences, list) or not sentences:
        return jsonify(({"error": "Request body must be a JSON list of sentences."}), 
                       400)

    # Set lang
    base_lang = request.args.get("src_lang", "eng_Latn")
    dest_lang = request.args.get("tgt_lang", "fra_Latn")

    # load model
    translation = load_model.translate(sentences=sentences,
                                       src_lang=base_lang,
                                       tgt_lang=dest_lang)
    return jsonify(translation)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})
