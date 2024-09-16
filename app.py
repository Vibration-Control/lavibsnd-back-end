from flask import Flask, request
from neutralizer_optimization import neutralizer_optimization


app = Flask(__name__)

@app.route('/optimizeNeutralizer', methods=['POST'])
def optimizeNeutralizer():
    json_payload = request.json
    result = neutralizer_optimization(json_payload)
    return result

if __name__ == '__main__':
    app.run(debug=True)
