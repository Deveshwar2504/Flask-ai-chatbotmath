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

# Chat bubble CSS
bubble_css = """
<style>
.user-bubble {
    background: rgba(99, 102, 241, 0.3);
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
}
.bot-bubble {
    background: rgba(168, 85, 247, 0.3);
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
}
.chat-container {
    display: flex;
    flex-direction: column;
}
.user-align { align-self: flex-end; }
.bot-align { align-self: flex-start; }
</style>
"""
st.markdown(bubble_css, unsafe_allow_html=True)

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
# Chat input box with white text
st.markdown("<style> input {color:white !important;} </style>", unsafe_allow_html=True)
user = st.text_input("You:", "", placeholder="Ask me anything about ML or calculate maths like 12*4")("You:", "", placeholder="Ask me anything about ML or calculate maths like 12*4")

# Send button placed under input box
if st.button("Send"):
    # Display user bubble
    st.markdown(f"<div class='chat-container'><div class='user-bubble user-align'>👤 {msg}</div></div>", unsafe_allow_html=True)
("Send"):("Send"):
    msg = user
    clean = clean_text(msg)

    # Math calculation
    if is_math_expression(clean):
        result = safe_eval(clean)
        st.markdown(f"<div class='chat-container'><div class='bot-bubble bot-align'>💬 {result}</div></div>", unsafe_allow_html=True):** {result}")
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
                    st.markdown(f"<div class='chat-container'><div class='bot-bubble bot-align'>💬 {random.choice(intent['responses'])}</div></div>", unsafe_allow_html=True) {random.choice(intent['responses'])}")
                    break
        else:
            st.markdown("<div class='chat-container'><div class='bot-bubble bot-align'>💬 I do not understand. Please rephrase.</div></div>", unsafe_allow_html=True):** I do not understand. Please rephrase.")
