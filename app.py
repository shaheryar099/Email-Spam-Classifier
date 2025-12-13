import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (safe)
nltk.download('stopwords')

# Initialize stemmer and stopwords
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

# Text preprocessing function
def trans_case(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)

    filtered_words = []
    for word in tokens:
        if word not in STOPWORDS:
            filtered_words.append(ps.stem(word))

    return ' '.join(filtered_words)

# Load model and vectorizer
tfid = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title('📧 Email Spam Classifier')

sms = st.text_area('Enter the message')

if st.button('Predict'):
    processed_text = trans_case(sms)
    vector_input = tfid.transform([processed_text])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error('🚨 Spam Message')
    else:
        st.success('✅ Not Spam')
