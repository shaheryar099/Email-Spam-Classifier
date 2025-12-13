import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def trans_case(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    x = []
    for i in tokens:
        if i.isalnum():
            x.append(i)

    y = []
    for i in x:
        if i not in stopwords.words('english'):
            y.append(i)

    z = []
    for i in y:
        z.append(ps.stem(i))

    return ' '.join(z)

tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email Spam Classifier')
sms = st.text_area('Enter Text', help='Enter SMS')

if st.button('Predict'):
    
    text_input = trans_case(sms)
    vector_input = tfid.transform([text_input])
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
    