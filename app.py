
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "artifacts/ecommerce_pipeline.joblib"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )
    return joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({
        "message": "Ecommerce High-Value Purchase Prediction API",
        "endpoint": "/predict",
        "method": "POST"
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": "Send a JSON object"}), 400

        model = load_model()
        features = pd.DataFrame([data])
        prediction = int(model.predict(features)[0])

        return jsonify({
            "high_value_purchase": prediction,
            "label": "High Value" if prediction == 1 else "Not High Value"
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
