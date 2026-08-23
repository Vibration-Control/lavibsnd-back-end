from flask import Flask, request, jsonify
from flask_cors import CORS
from neutralizer_optimization import neutralizer_optimization
from services.rst_to_json import convert_rst_to_json

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route("/optimizeNeutralizer", methods=["POST"])
def optimizeNeutralizer():
    json_payload = request.json
    result = neutralizer_optimization(json_payload)
    return jsonify(result)


@app.route("/convertRst", methods=["POST"])
def convertRst():
    if "file" not in request.files:
        return jsonify({"error": "Missing .rst file"}), 400

    rst_file = request.files["file"]

    try:
        json_result = convert_rst_to_json(rst_file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(json_result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
