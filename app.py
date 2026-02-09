import streamlit as st
import json
import torch
import random
from model import NeuralNet
from utils import tokenize, bag_of_words, clean_text, is_math_expression, safe_eval

# Page config
st.set_page_config(page_title="AI Chatbot", layout="centered")

# Background style
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    background-size: cover;
}}
[data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title UI
st.markdown("<h1 style='text-align:center;color:white;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Load ML Model
FILE = "model.pth"
data = torch.load(FILE, map_location=torch.device("cpu"))
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Chat input box
user = st.text_input("You:", "", placeholder="Ask me anything about ML or calculate maths like 12*4")

if st.button("Send"):
    msg = user
    clean = clean_text(msg)

    # Math calculation
    if is_math_expression(clean):
        result = safe_eval(clean)
        st.write(f"**Bot:** {result}")
    else:
        # NLP model
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = torch.from_numpy(X).unsqueeze(0)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    st.write(f"**Bot:** {random.choice(intent['responses'])}")
                    break
        else:
            st.write("**Bot:** I do not understand. Please rephrase.")
