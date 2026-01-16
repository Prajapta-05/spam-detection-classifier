import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    # Lower case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # remove special characters
    txt = []
    for i in text:
        if i.isalnum():
            txt.append(i)

    # remove stopwords and punctuations
    text = txt.copy()
    txt.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            txt.append(i)

    # stemming
    text = txt.copy()
    txt.clear()

    for i in text:
        txt.append(ps.stem(i))

    return " ".join(txt)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS SPAM CLASSIFIER")

input_sms=st.text_area("Enter the message to check whether it is a spam message or not.")


if st.button('PREDICT'):
    # here we have to follow 4 steps
    #1. preprocess
    transformed_sms = transform_text(input_sms)

    #2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    #3. predict
    result = model.predict(vector_input)[0]

    #4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
