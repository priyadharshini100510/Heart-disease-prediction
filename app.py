from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging

app = Flask(__name__)

# Load the trained model
with open('ensemble_model.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    logging.info(f"Received data: {data}")
    # Convert input data to DataFrame or appropriate format
    input_data = pd.DataFrame(data)
    prediction = ensemble_model.predict(input_data)
    logging.info(f"Prediction: {prediction.tolist()}")
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)