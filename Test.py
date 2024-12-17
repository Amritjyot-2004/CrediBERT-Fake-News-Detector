import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

model_path = "D:/Projects/FND/Detector"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

def classify_news(headline):
    
    inputs = tokenizer(headline, return_tensors = 'pt', padding = True, truncation = True, max_length = 32)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

news_map = {0:"Real", 1:"Fake"}

st.title("\t\tCrediBERT\n\t\tYour Personalised Fake News Detector")
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h2 class="title">Insert the news headline to validate its authenticity</h1>', unsafe_allow_html=True)

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt = st.chat_input("Insert News Headline")

if prompt:
    # Display user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add the user's message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = classify_news(prompt)
        assistant_message = f"This news is {news_map[response]}"
        st.markdown(assistant_message)

    # Append assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})