import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load models
MODELS = {
    'CNN': load_model('sentiment_analysis_cnn.keras'),
    'LSTM': load_model('sentiment_analysis_lstm.keras')
}

max_len = 100  # Must match training

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis App")
st.write("Choose a model and enter some text to analyze the sentiment.")

# User input
text_input = st.text_area("Enter text", height=150)
model_choice = st.selectbox("Select Model", list(MODELS.keys()))

if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        model = MODELS[model_choice]
        sequence = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(sequence, maxlen=max_len)
        probability = float(model.predict(padded)[0][0])
        label = "Positive" if probability >= 0.5 else "Negative"

        st.subheader("Result")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {round(probability * 100, 2)}%")
        st.write(f"**Model Used:** {model_choice}")
