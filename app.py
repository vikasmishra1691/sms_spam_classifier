import streamlit as st
import pickle
import string
import nltk
import os

# Set a specific directory for NLTK data that Render can write to
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Always download resources at startup
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
          y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS spam classifier")

input_text = st.text_area("Enter your message")

if st.button("predict"):
    # Preprocess
    transformed_text = transform_text(input_text)
    # Vectorize
    vectorized_message = tfidf.transform([transformed_text])
    # Predict
    result = model.predict(vectorized_message)[0]
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")