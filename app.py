from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download NLTK stopwords
nltk.download('stopwords')

app = Flask(__name__)

# Load the saved SVC model
svc_model = joblib.load('svc_model.pkl')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Define the preprocessing
def preprocess_text(text):
    # Removing all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Converting all text to lowercase
    text = text.lower()
    # Removing stopwords from text
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Apply stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

@app.route('/')
def home():
    return "Welcome to the AirCrashPrediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Preprocess the 'Report' and 'Part Failure' text
    report = preprocess_text(data['Report'].strip().lower())
    part_failure = preprocess_text(data['Part Failure'].strip().lower())
    
    # Combine the preprocessed text for prediction
    input_data = pd.DataFrame({
        'Report': [report],
        'Part Failure': [part_failure]
    })
    
    # prediction
    prediction = svc_model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
