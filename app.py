# %%% Start of app.py - keyword-based prediction version

import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Hardcoded keywords for detection ---
fake_keywords = [
    "lottery", "cash prize", "win", "free", "claim now", 
    "click here", "urgent", "rich quick", "bonus money", "miracle cure",
    "bitcoin", "luxury car", "iPhones", "double your bank", "investment scheme",
    "exclusive", "instant reward", "get rich", "prize", "register today"
]

true_keywords = [
    "government", "announces", "policy", "rbi", "nasa", "scientists",
    "supreme court", "budget", "health initiative", "pm", "inaugurates",
    "election", "guidelines", "infrastructure", "education", "research",
    "growth", "report", "update", "official"
]

# --- Prediction Function ---
def predict_news(text):
    text_lower = text.lower()
    
    # Check fake keywords first
    if any(word in text_lower for word in fake_keywords):
        return "Fake"
    # Check true keywords
    if any(word in text_lower for word in true_keywords):
        return "True"
    
    # Otherwise, fallback to trained model
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)[0][0]
    return "Fake" if prediction > 0.5 else "True"

# --- Streamlit UI ---
st.title("Fake vs True News Detector")
st.write(
    "Enter any news text below and the app will predict whether it is **Fake** or **True** "
    "based on keywords and trained model."
)

# --- User Input for Prediction ---
user_input = st.text_area("Enter your news text here:")
if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_news(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")

