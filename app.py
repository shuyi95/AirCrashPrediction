from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re

app = Flask(__name__)

# Load the saved SVC model
svc_model = joblib.load('svc_model.pkl')

# Define preprocessing function
def preprocess_text(text):
    # preprocessing
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Preprocess the 'Report' text
    report = preprocess_text(data['Report'])
    
    # Combine the preprocessed text with 'Part Failure'
    input_data = pd.DataFrame({
        'Report': [report],
        'Part Failure': [data['Part Failure']]
    })
    
    # Make prediction
    prediction = svc_model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
