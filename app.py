# %%% Start of app.py - Fake vs True News Detector with Confidence Score

import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Streamlit UI ---
st.title("Fake vs True News Detector")
st.write(
    "This app predicts whether a news article is **Fake** or **True** "
    "and shows the confidence score."
)

# --- Prediction Function ---
def predict_news(text):
    if not text.strip():
        return "No input provided", 0.0
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    result = "Fake" if prediction > 0.5 else "True"
    return result, confidence

# --- User Input for Prediction ---
st.subheader("Try it Yourself")
user_input = st.text_area("Enter your news text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        result, confidence = predict_news(user_input)
        st.success(f"Prediction: {result}")
        st.info(f"Confidence Score: {confidence*100:.2f}%")
    else:
        st.warning("Please enter some text.")

