# --- app.py: Comprehensive Fake vs True News Detector ---

# app.py - Comprehensive Fake vs True News Detector

import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Fake News Examples (for model context, not UI) ---
fake_news_examples = [
    "Congratulations! You have been randomly selected to receive ₹10,00,000 from the government’s secret lottery program; just click this link and enter your bank details to claim your cash prize immediately.",
    "Earn ₹50,000 per day from home without any investment, just follow this secret method and double your money overnight.",
    "Miracle cure for diabetes discovered, take this pill today and lose 20 kg in 2 weeks!",
    "Government is giving free iPhones to random citizens, apply now to receive yours.",
    "Secret trick to double your bank balance overnight using this hidden technique.",
    "You are selected for free luxury car giveaway, claim now before it ends!",
    "Bitcoin investment scheme promises 100% return in 1 week, join today!",
    "Celebrity endorses weight-loss pill that burns fat instantly, buy now!",
    "Covid vaccine causes infertility, warn scientists secretly, read now!",
    "Banks will forgive all loans for select citizens, click here to know if you qualify."
]

# --- True News Examples (for model context, not UI) ---
true_news_examples = [
    "India's GDP grows by 6% in the last fiscal year according to the latest report by the Reserve Bank of India, reflecting steady economic growth across multiple sectors.",
    "The Indian government announced a new nationwide digital health initiative today, aiming to provide affordable healthcare services to millions of citizens.",
    "NASA confirms the discovery of a new exoplanet in the habitable zone suitable for future exploration.",
    "Supreme Court of India passed a verdict today regarding a major property dispute, clarifying legal interpretations.",
    "The Ministry of Education releases updated guidelines for schools to implement innovative teaching methods nationwide.",
    "PM inaugurates a new highway connecting multiple states, improving trade and travel efficiency.",
    "Monsoon rains expected in coastal Maharashtra this week, authorities advise citizens to take precautions.",
    "Indian cricket team wins ODI series against Australia, players praised for outstanding performance.",
    "Election Commission releases updated voter ID rules to improve election transparency and accessibility.",
    "State government increases budget allocation for schools and healthcare to support community welfare."
]

# --- Prediction Function ---
def predict_news(text):
    if not text.strip():
        return "No input provided", 0.0
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)[0][0]
    label = "Fake" if prediction > 0.5 else "True"
    confidence = prediction if label == "Fake" else 1 - prediction
    confidence_percent = round(confidence * 100, 2)
    return label, confidence_percent

# --- Streamlit UI ---
st.title("Fake vs True News Detector")
st.write(
    "This app predicts whether a news article is **Fake** or **True** using a trained LSTM model."
)

st.subheader("Try it Yourself")
user_input = st.text_area("Enter your news text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        label, confidence = predict_news(user_input)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence Score: {confidence}%")
    else:
        st.warning("Please enter some text to predict.")


