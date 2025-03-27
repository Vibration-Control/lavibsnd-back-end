from flask import Flask, request, jsonify
from flask_cors import CORS
from neutralizer_optimization import neutralizer_optimization

app = Flask(__name__)
# Allow all origins
CORS(app, supports_credentials=True)

@app.route('/optimizeNeutralizer', methods=['POST'])
def optimizeNeutralizer():
    json_payload = request.json
    result = neutralizer_optimization(json_payload)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)