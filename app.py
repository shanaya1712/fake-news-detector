import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Prediction function
def predict_news(text):
    if not text.strip():
        return "No input provided", 0.0
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]
    label = "Fake" if pred > 0.5 else "True"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence*100

# Streamlit UI
st.title("Fake vs True News Detector")
st.write("Enter your news text below to check if it is Fake or True:")

user_input = st.text_area("Enter your news text here:")
if st.button("Predict"):
    label, confidence = predict_news(user_input)
    st.success(f"Prediction: {label} | Confidence: {confidence:.2f}%")
