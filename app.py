import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
MODEL_NAME = "unitary/toxic-bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.sigmoid(logits).squeeze().tolist()
    
    threshold = 0.5  # Adjust as needed
    return "Toxic" if prediction[0] > threshold else "Not Toxic"

# Streamlit UI
st.title("Toxic Speech Classifier")
st.write("Enter a sentence below to classify whether it's toxic or not.")

user_input = st.text_area("Enter text:", "")

if st.button("Classify"):
    if user_input.strip():
        result = classify_text(user_input)
        st.write(f"**Prediction:** {result}")
    else:
        st.warning("Please enter some text to classify.")
