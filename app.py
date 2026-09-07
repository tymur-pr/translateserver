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

    # Set language
    base_lang = request.args.get("src_lang", "en")
    dest_lang = request.args.get("tgt_lang")

    if not dest_lang:
        return jsonify({"error": "tgt_lang query parameter is required."}), 400

    try:
        translation = load_model.translate(sentences=sentences,
                                           src_lang=base_lang,
                                           tgt_lang=dest_lang)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    response = jsonify(translation)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})
