import streamlit as st
import json
import torch
import random
from model import NeuralNet
from utils import tokenize, bag_of_words, clean_text, is_math_expression, safe_eval

st.set_page_config(page_title="AI Chatbot", layout="centered")

# Background + Chat Bubble CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #4f46e5, #9333ea);
}

[data-testid="stHeader"] {background: rgba(0,0,0,0);}

/* Chat Window */
.chat-window {
    height: 420px;
    overflow-y: auto;
    padding: 10px;
    display: flex;
    flex-direction: column;
}

/* Chat Bubbles */
.user-bubble {
    background: rgba(99, 102, 241, 0.35);
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
    align-self: flex-end;
}

.bot-bubble {
    background: rgba(168, 85, 247, 0.35);
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
    align-self: flex-start;
}

/* Input text black */
input, textarea {color: black !important;}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:white;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)

# Session state for chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Load ML model
data = torch.load("model.pth", map_location=torch.device("cpu"))
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()
all_words = data["all_words"]
tags = data["tags"]

# CHAT WINDOW
st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
for role, text in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>👤 {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>💬 {text}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# INPUT BAR BELOW CHAT
txt = st.text_input("You:", "", placeholder="Ask ML questions or type maths like 12*4")
if st.button("Send") and txt.strip() != "":

    st.session_state.chat.append(["user", txt])
    clean = clean_text(txt)

    # Math calculation
    if is_math_expression(clean):
        reply = str(safe_eval(clean))
        st.session_state.chat.append(["bot", reply])
        st.experimental_rerun()

    # NLP inference
    sentence = tokenize(txt)
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
                st.session_state.chat.append(["bot", random.choice(intent["responses"])])
                break
    else:
        st.session_state.chat.append(["bot", "I do not understand. Please rephrase."])

    st.experimental_rerun()


