import streamlit as st
import pickle
import os
import nltk
from preprocessing import clean # cite: 1

# --- STEP A: DOWNLOAD NLTK DATA ---
# This fixes the "LookupError" common in text processing apps
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

# --- STEP B: SET UP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_email_model.pkl")

# --- STEP C: LOAD MODEL ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file) # cite: 2

model = load_my_model()

# --- STEP D: THE INTERFACE ---
st.title("📧 Spam Email Classifier")

if model is None:
    st.error(f"Error: Could not find 'spam_email_model.pkl' in {BASE_DIR}")
    st.stop()

email_input = st.text_area("Paste email text here:", height=150)

if st.button("Check Email"):
    if email_input.strip():
        # The model uses the 'clean' function automatically inside the Pipeline
        prediction = model.predict([email_input])[0] # cite: 2
        
        if prediction.lower() == "spam":
            st.error("### 🚨 Result: This is SPAM")
        else:
            st.success("### ✅ Result: This is HAM (Safe)")
    else:
        st.warning("Please enter text to analyze.")