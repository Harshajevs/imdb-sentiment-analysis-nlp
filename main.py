# app/main.py
import streamlit as st
import tensorflow as tf
import json
import re
from bs4 import BeautifulSoup
import os
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom preprocessing class to match the original notebook
class CustomPreprocess:
    def __init__(self):
        pass

    def preprocess_text(self, sen):
        sen = sen.lower()
        
        # Remove html tags
        sentence = self.remove_tags(sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Remove multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Remove Stopwords
        from nltk.corpus import stopwords
        stopwords_list = set(stopwords.words('english'))
        pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
        sentence = pattern.sub('', sentence)
        
        return sentence
    
    def remove_tags(self, text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

# Load the trained model and tokenizer with caching
@st.cache_resource
def load_components():
    import os
    
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'sentiment_classifier.h5')
    tokenizer_path = os.path.join(base_dir, 'models', 'tokenizer.json')
    
    # Debug information
    print(f"Looking for model at: {model_path}")
    print(f"Looking for tokenizer at: {tokenizer_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Tokenizer exists: {os.path.exists(tokenizer_path)}")
    
    # Load model with tf.keras
    model = tf.keras.models.load_model(model_path)
    
    # Load tokenizer from JSON
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    
    return model, tokenizer, CustomPreprocess()

# Load model and tokenizer at the beginning
try:
    model, tokenizer, preprocessor = load_components()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None
    tokenizer = None
    preprocessor = None

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analyzer 🎬")
st.write("Enter a movie review below to predict its sentiment:")

review_input = st.text_area("Your Review:", 
                            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.")

if st.button("Analyze"):
    if not review_input.strip():
        st.error("Please enter a review to analyze.")
    elif model is None or tokenizer is None:
        st.error("Model or tokenizer failed to load. Please check the error message above.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess input exactly like in the notebook
            cleaned = preprocessor.preprocess_text(review_input)
            
            # Tokenize and pad
            seq = tokenizer.texts_to_sequences([cleaned])
            padded_seq = pad_sequences(seq, padding='post', maxlen=100)
            
            # Get raw prediction
            raw_pred = model.predict(padded_seq)[0][0]
            
            # Scale to 0-10 like in the notebook
            scaled_pred = round(float(raw_pred * 10), 1)
            
            # Determine sentiment based on the 5.0 threshold
            sentiment = "Positive 😊" if scaled_pred > 5 else "Negative 😠"
            
            # Calculate confidence based on distance from threshold
            if scaled_pred > 5:
                confidence = (scaled_pred - 5) / 5  # Scale to 0-100%
            else:
                confidence = (5 - scaled_pred) / 5
            confidence = min(confidence, 1.0)  # Cap at 100%
            
            # Display results
            st.subheader("Prediction Result:")
            st.metric("Rating (0-10)", f"{scaled_pred}")
            st.metric("Sentiment", sentiment)
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Debug information
            st.write("**Raw model output:**", raw_pred)
            st.write("**Processed Review:**", cleaned)

st.markdown("---")
st.write("Powered by TensorFlow & Streamlit | Model trained on IMDB dataset")

# To include CSS styling
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # Silently continue if the CSS file doesn't exist
    pass
