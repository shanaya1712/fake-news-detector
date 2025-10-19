# app.py - updated version with fixed buttons and demo functionality

import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Fake News Examples with Real-world Keywords ---
fake_news_examples = [
    "You won a lottery of ₹5,00,000! Click here to claim your cash prize.",
    "Earn ₹50,000 per day from home without any investment!",
    "Miracle cure for diabetes discovered, take this pill today!",
    "Government giving free iPhones to random citizens, apply now!",
    "Secret trick to double your bank balance overnight!",
    "You are selected for free luxury car giveaway, claim now!",
    "Bitcoin investment scheme promises 100% return in 1 week!",
    "Celebrity endorses weight-loss pill that burns fat instantly!",
    "Covid vaccine causes infertility, warn scientists!",
    "Banks will forgive all loans for select citizens, click here!",
    "Win free gold coins just by sharing this link!",
    "Exclusive secret to become rich overnight without work!",
    "Magic oil that makes you invisible discovered, buy now!",
    "Free holiday trip abroad for random participants, register today!",
    "Government will pay you ₹10,00,000 if you participate in survey!",
]

# --- True News Examples with Real-world Keywords ---
true_news_examples = [
    "Indian government announces new education policy from 2026.",
    "Stock market rises by 200 points after RBI announcement.",
    "NASA confirms discovery of new exoplanet in habitable zone.",
    "Mumbai receives heavy rainfall, traffic disrupted in city.",
    "Supreme Court rules on property dispute, verdict released.",
    "WHO releases updated guidelines on COVID-19 vaccination.",
    "India's GDP grows by 6% in the last fiscal year, reports RBI.",
    "PM inaugurates new highway connecting multiple states.",
    "Scientists discover rare species of bird in Western Ghats.",
    "Election commission releases updated voter ID rules.",
    "Government launches new digital health initiative nationwide.",
    "State government increases budget allocation for schools.",
    "Indian cricket team wins series against Australia in ODI.",
    "Monsoon rains expected in coastal Maharashtra this week.",
    "Union Budget announced with focus on infrastructure growth.",
]

# --- Streamlit UI ---
st.title("Fake vs True News Detector")
st.write(
    "This app predicts whether a news article is **Fake** or **True** "
    "based on trained model keywords and real-world examples."
)

# --- Demo Buttons Using Columns to Avoid Conflict ---
col1, col2 = st.columns(2)

with col1:
    if st.button("Show Fake News Demo"):
        st.subheader("Fake News Examples")
        for news in fake_news_examples:
            st.write("- ", news)

with col2:
    if st.button("Show True News Demo"):
        st.subheader("True News Examples")
        for news in true_news_examples:
            st.write("- ", news)

# --- Prediction Function ---
def predict_news(text):
    if not text.strip():
        return "No input provided"
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)[0][0]
    return "Fake" if prediction > 0.5 else "True"

# --- User Input for Prediction ---
st.subheader("Try it Yourself")
user_input = st.text_area("Enter your news text here:")

if st.button("Predict News"):
    if user_input.strip():
        result = predict_news(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")
