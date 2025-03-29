from flask import Flask, request, jsonify

import mlflow
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
app = Flask(__name__)

handler = RotatingFileHandler('wine_api.log', maxBytes=10000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

MLFLOW_TRACKING_URI = 'http://localhost:5000'
MODEL_URI = 'models:/WineApp/Production'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)
app.logger.info(f"Model loaded {MODEL_URI}")

REQUIRED_FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]


@app.route('/predict', methods=['POST'])
def predict():
        data = request.json
        if not all(feat in data for feat in REQUIRED_FEATURES):
            missing = [feat for feat in REQUIRED_FEATURES if feat not in data]
            app.logger.warning(f"Missing features: {missing}")
            return jsonify({"error": f"Missing required features: {missing}"}), 400


        features = np.array([float(data[feat]) for feat in REQUIRED_FEATURES]).reshape(1, -1)
        features_scaled = scaler.transform(features)


        prediction = model.predict(features_scaled)
        quality = int(prediction[0])

        app.logger.info(f"Prediction successful - quality: {quality}")
        return jsonify({
            "quality": quality,
            "quality_label": f"{quality}/10",
            "input_features": data
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# curl -X POST http:http://127.0.0.1/predict -H "Content-Type:application/json" -d '{}'