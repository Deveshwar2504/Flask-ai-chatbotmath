import re
import numpy as np


def tokenize(sentence):
    # simple, dependency-free tokenizer: split on word boundaries
    return re.findall(r"\b\w+\b", sentence.lower())


def stem(word):
    w = word.lower()
    # light-weight suffix stripping to approximate stemming
    for suf in ("ing", "ly", "ed", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if stem(w) in tokenized_sentence:
            bag[idx] = 1.0
    return bag

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9+\-*/%. ]', '', text)
    return text.strip()


def is_math_expression(text):
    return bool(re.search(r'[0-9+\-*/%]', text))


def safe_eval(expr):
    try:
        if '%' in expr and 'of' in expr:
            p = expr.split('% of ')
            return float(p[0]) * float(p[1]) / 100
        return eval(expr, {"__builtins__": None}, {})
    except:
        return "Invalid calculation"
