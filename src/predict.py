# ==========================================================
# SMARTBRIDGE AI LEARNING ASSISTANT
# TASK 6 : EMOTION PREDICTION
# ==========================================================

import re
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.model import focal_loss

# ==========================================================
# CONFIGURATION
# ==========================================================

MAX_SEQUENCE_LENGTH = 80

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading Adaptive BiLSTM Model...")

model = load_model(

    "models/bilstm_student_adaptive.keras",

    custom_objects={
        "loss": focal_loss()
    }

)

try:
    print("✅ Model Loaded")
except UnicodeEncodeError:
    print("[SUCCESS] Model Loaded")

# ==========================================================
# LOAD TOKENIZER
# ==========================================================

with open(

    "data/processed/tokenizer.pkl",

    "rb"

) as file:

    tokenizer = pickle.load(file)

# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

with open(

    "data/processed/label_encoder.pkl",

    "rb"

) as file:

    label_encoder = pickle.load(file)

# ==========================================================
# LOAD RESPONSES
# ==========================================================

responses = pd.read_csv("emotion_response_mapping.csv")

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z!? ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_emotion(text):

    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(

        sequence,

        maxlen=MAX_SEQUENCE_LENGTH,

        padding="post",

        truncating="post"

    )

    prediction = model.predict(padded, verbose=0)
    print("\nRaw prediction:", prediction)
    print("Probabilities:", prediction[0])
    probabilities = prediction[0]

    predicted_index = np.argmax(probabilities)

    confidence = float(np.max(probabilities))

    emotion = label_encoder.inverse_transform(
          [predicted_index]
    )[0]

    advice = responses.loc[

        responses["Emotion"] == emotion,

        "Advice"

    ].values[0]

    return (
    emotion,
    advice,
    confidence,
    probabilities,
    label_encoder.classes_
)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\nEmotion Detection Ready!")
    print("Type 'exit' to quit.\n")

    while True:

        text = input("Enter Student Text : ")

        if text.lower() == "exit":

            print("\nGoodbye!")

            break

        emotion, advice, confidence, probabilities, class_names = predict_emotion(text)
        print("\nEmotion :", emotion)

        print("Advice  :", advice)

        print("-" * 60)