# --- app.py: Improved Fake vs True News Detector ---
import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Internal Keywords for Reference (Developer Only) ---
FAKE_KEYWORDS = [
    "lottery", "cash prize", "click here", "win", "selected", "free",
    "miracle cure", "overnight", "bitcoin investment", "double your money",
    "exclusive", "secret trick", "luxury car", "gold coins", "magic", "buy now"
]

TRUE_KEYWORDS = [
    "government", "policy", "announcement", "rbi", "reports", "inaugurates",
    "scientists", "discovery", "guidelines", "budget", "election", "GDP",
    "cricket", "weather", "supreme court", "digital initiative"
]

# --- Text Preprocessing Function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

# --- Prediction Function ---
def predict_news(text):
    if not text.strip():
        return "No input provided", 0.0
    
    clean_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=200)
    
    prediction_prob = model.predict(padded)[0][0]
    
    # Dynamic threshold for Fake detection
    threshold = 0.7
    if prediction_prob > threshold:
        label = "Fake"
        confidence = prediction_prob
    else:
        label = "True"
        confidence = 1 - prediction_prob
    
    return label, confidence

# --- Streamlit UI ---
st.title("Fake vs True News Detector")
st.write(
    "This app predicts whether a news article is **Fake** or **True** "
    "using a trained LSTM model. Enter your news text below to try it out."
)

st.subheader("Try it Yourself")
user_input = st.text_area("Enter your news text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        label, confidence = predict_news(user_input)
        st.success(f"Prediction: {label} | Confidence: {confidence*100:.2f}%")
    else:
        st.warning("Please enter some text.")

# Optional: you can include internal reference examples for testing (not shown in UI)
# FAKE_EXAMPLES = ["You won a lottery!", "Claim your free iPhone now!"]
# TRUE_EXAMPLES = ["India reports 6% GDP growth.", "Government launches digital health initiative."]



