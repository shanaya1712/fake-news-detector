import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model('fake_news_model.h5', compile=False)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Streamlit UI ---
st.title("Fake News Detection")

# Fake news examples with keywords
fake_examples = [
    "You won a lottery! Claim your $1,000,000 cash prize now!",
    "Breaking: Scientists discover miracle cure for diabetes, no prescription needed!",
    "Get rich quick! Invest $100 today and earn $10,000 tomorrow!",
    "Urgent: Your bank account is at risk! Click this link to secure your funds!",
    "Celebrity secret revealed: This one trick makes you lose 20kg in a week!"
]

# True news examples
true_examples = [
    "The Prime Minister inaugurated a new highway project in Delhi today.",
    "Scientists published a new study on climate change in Nature journal.",
    "Local authorities announced new safety measures for city schools.",
    "The annual budget report was released by the finance ministry.",
    "A new vaccination program was launched to prevent seasonal flu."
]

# Buttons for examples
st.write("### Demo News Examples")
if st.button("Random Fake News Example"):
    news_text = random.choice(fake_examples)
elif st.button("Random True News Example"):
    news_text = random.choice(true_examples)
else:
    news_text = st.text_area("Enter news text here:")

# Predict button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text first!")
    else:
        seq = tokenizer.texts_to_sequences([news_text])
        padded = pad_sequences(seq, maxlen=100)  # match training maxlen
        pred = model.predict(padded)[0][0]
        label = "Fake" if pred > 0.5 else "True"
        st.write(f"Prediction: {label} (Score: {pred:.4f})")
        st.write("### News Text:")
        st.write(news_text)
