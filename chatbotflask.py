from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from utils import tokenize, bag_of_words, clean_text, is_math_expression, safe_eval

app = Flask(__name__)

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Load trained model
FILE = "model.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))  # use CPU for Render free
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

@app.route("/")
def home():
    return render_template("aichatbot.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    msg = request.json.get("msg")
    clean = clean_text(msg)

    # Maths calculation check
    if is_math_expression(clean):
        return jsonify({"reply": str(safe_eval(clean))})

    # Chatbot NLP response
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
                return jsonify({"reply": random.choice(intent["responses"])})
    return jsonify({"reply": "I do not understand. Please rephrase."})


