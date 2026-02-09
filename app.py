import streamlit as st
import json
import torch
import random
from model import NeuralNet
from utils import tokenize, bag_of_words, clean_text, is_math_expression, safe_eval

# Page configuration
st.set_page_config(page_title="AI Chatbot", layout="centered")

# Background gradient
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #4f46e5, #9333ea);
}

[data-testid="stHeader"] {background: rgba(0,0,0,0);}

.user-bubble {
    background: rgba(99, 102, 241, 0.35);
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
}

.bot-bubble {
    background: rgba(168, 85, 247, 0.35);
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
}

.user-align {align-self: flex-end;}
.bot-align {align-self: flex-start;}

.chat-container {
    display: flex;
    flex-direction: column;
}
input {color:white !important;}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;color:white;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Load ML model
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

# Chat input
msg = st.text_input("You:", "", placeholder="Ask something about Machine Learning or type maths like 12*4")

# SEND button
if st.button("Send"):

    # user bubble
    st.markdown(
        f"""
        <div class='chat-container'>
            <div class='user-bubble user-align'>👤 {msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    clean = clean_text(msg)

    # Math calculator
    if is_math_expression(clean):
        result = safe_eval(clean)
        st.markdown(
            f"""
            <div class='chat-container'>
                <div class='bot-bubble bot-align'>💬 {result}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                    reply = random.choice(intent["responses"])
                    st.markdown(
                        f"""
                        <div class='chat-container'>
                            <div class='bot-bubble bot-align'>💬 {reply}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    break
        else:
            st.markdown(
                """
                <div class='chat-container'>
                    <div class='bot-bubble bot-align'>💬 I do not understand. Please rephrase.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
