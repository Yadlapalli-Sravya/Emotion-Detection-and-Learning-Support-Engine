# ==========================================================
# SMARTBRIDGE AI LEARNING ASSISTANT
# TASK 5 : BERT MODEL FINE-TUNING
# ==========================================================

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification
)

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 80
NUM_CLASSES = 4

os.makedirs("models/bert_emotion_model", exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

train_df = pd.read_csv("data/raw/train.csv")
val_df = pd.read_csv("data/raw/validation.csv")

# ==========================================================
# LABEL MAPPING
# ==========================================================

emotion_map = {

    0: "Frustrated",
    1: "Confident",
    2: "Confident",
    3: "Frustrated",
    4: "Confused",
    5: "Curious"

}

train_df["emotion"] = train_df["label"].map(emotion_map)
val_df["emotion"] = val_df["label"].map(emotion_map)

encoder = LabelEncoder()

encoder.fit(train_df["emotion"])

y_train = encoder.transform(train_df["emotion"])
y_val = encoder.transform(val_df["emotion"])

with open("data/processed/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# ==========================================================
# TOKENIZER
# ==========================================================

print("\nLoading BERT Tokenizer...")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(

    train_df["text"].tolist(),

    truncation=True,

    padding=True,

    max_length=MAX_LENGTH,

    return_tensors="tf"

)

val_encodings = tokenizer(

    val_df["text"].tolist(),

    truncation=True,

    padding=True,

    max_length=MAX_LENGTH,

    return_tensors="tf"

)

# ==========================================================
# MODEL
# ==========================================================

print("\nLoading BERT Model...")

model = TFBertForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=NUM_CLASSES

)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(

    optimizer=optimizer,

    loss=loss,

    metrics=["accuracy"]

)

# ==========================================================
# TRAIN
# ==========================================================

print("\nStarting Fine-Tuning...\n")

history = model.fit(

    {

        "input_ids": train_encodings["input_ids"],

        "attention_mask": train_encodings["attention_mask"]

    },

    y_train,

    validation_data=(

        {

            "input_ids": val_encodings["input_ids"],

            "attention_mask": val_encodings["attention_mask"]

        },

        y_val

    ),

    epochs=3,

    batch_size=16

)

# ==========================================================
# SAVE MODEL
# ==========================================================

print("\nSaving Model...")

model.save_pretrained("models/bert_emotion_model")

tokenizer.save_pretrained("models/bert_emotion_model")

print("\nModel Saved Successfully!")

print("Location : models/bert_emotion_model")

# ==========================================================
# FINAL RESULTS
# ==========================================================

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Training Accuracy   : {train_acc:.4f}")
print(f"Validation Accuracy : {val_acc:.4f}")

print("=" * 60)