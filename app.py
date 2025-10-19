# --- app.py: Comprehensive Fake vs True News Detector ---

import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('fake_news_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- Keywords for internal reference (no buttons) ---
fake_keywords = [
    "lottery", "cash prize", "click here", "win", "selected", "free", 
    "miracle cure", "overnight", "bitcoin investment", "double your money",
    "exclusive", "secret trick", "luxury car", "gold coins", "magic", "buy now"
]

true_keywords = [
    "government", "policy", "announcement", "rbi", "reports", "inaugurates",
    "scientists", "discovery", "guidelines", "budget", "election", 
    "GDP", "cricket", "weather", "supreme court", "digital initiative"
]

# --- Streamlit UI ---
st.set_page_config(page_title="Fake vs True News Detector", page_icon="📰")
st.title("📰 Fake vs True News Detector with Confidence Score")
st.write(
    "This app predicts whether a news article is **Fake** or **True** "
    "based on a trained model and real-world keyword examples."
)

# --- Prediction Function ---
def predict_news(text):
    """
    Predict if a news text is Fake or True and return confidence score.
    """
    if not text.strip():
        return "No input provided", 0.0

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)

    # Predict
    prediction = model.predict(padded)[0][0]

    # Determine result
    result = "Fake" if prediction > 0.5 else "True"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence_percent = round(confidence * 100, 2)

    return result, confidence_percent

# --- User Input for Prediction ---
st.subheader("Try it Yourself")
user_input = st.text_area("Enter your news text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        result, confidence = predict_news(user_input)
        # Color code result
        if result == "Fake":
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")
        st.info(f"Confidence Score: {confidence}%")
    else:
        st.warning("Please enter some text to predict.")

# --- Optional: Example Long News ---
st.subheader("Example Sentences to Test")
st.markdown("""
**Fake News Example:**  
"Congratulations! You have been randomly selected to receive ₹10,00,000 from the government’s secret lottery program; just click this link and enter your bank details to claim your cash prize immediately."

**True News Example:**  
"India's GDP grows by 6% in the last fiscal year according to the latest report by the Reserve Bank of India, reflecting steady economic growth across multiple sectors."
""")

# --- Internal Keyword Display (Optional, for debugging) ---
st.subheader("Internal Keyword References (for developer)")
st.write("Fake Keywords:", fake_keywords)
st.write("True Keywords:", true_keywords)

