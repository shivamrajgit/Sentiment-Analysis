import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 100  # consistent with training

# Pre-load both models for performance (or load on demand)
MODELS = {
    'cnn': load_model('sentiment_analysis_cnn.keras'),
    'lstm': load_model('sentiment_analysis_lstm.keras')
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    model_choice = request.form['model']

    if model_choice not in MODELS:
        return jsonify({'error': 'Invalid model choice'}), 400

    model = MODELS[model_choice]

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    probability = float(model.predict(padded)[0][0])
    label = 'Positive' if probability >= 0.5 else 'Negative'

    return jsonify({
        'label': label,
        'probability': round(probability * 100, 2),
        'model': model_choice.upper()
    })

if __name__ == '__main__':
    app.run(debug=True)
