import streamlit as st
import json
import torch
import random
from model import NeuralNet
from utils import tokenize, bag_of_words, clean_text, is_math_expression, safe_eval

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Load model
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

# UI
st.title("🤖 AI Chatbot")

# Chat input
user_input = st.text_input("You:", "")

if st.button("Send"):
    msg = user_input
    clean = clean_text(msg)

    # Math
    if is_math_expression(clean):
        st.write("Bot:", safe_eval(clean))
    else:
        # NLP response
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
                    st.write("Bot:", random.choice(intent["responses"]))
                    break
        else:
            st.write("Bot: I do not understand. Please rephrase.")

