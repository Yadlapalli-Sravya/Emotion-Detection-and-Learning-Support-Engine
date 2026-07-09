# ==========================================================
# SMARTBRIDGE AI LEARNING ASSISTANT
# TASK 2 : DATA PREPROCESSING & TOKENIZATION
# ==========================================================

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

try:
    # Prefer tensorflow.keras preprocessing utilities
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    try:
        # Fallback to standalone Keras if tensorflow.keras is not available
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
    except Exception as e:
        raise ImportError(
            "Neither tensorflow.keras nor keras preprocessing is available. Install TensorFlow (e.g., pip install tensorflow) or Keras (pip install keras)."
        ) from e

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80

RAW_PATH = "data/raw"
SAVE_PATH = "data/processed"

os.makedirs(SAVE_PATH, exist_ok=True)

# ==========================================================
# LOAD DATASETS
# ==========================================================

print("="*60)
print("Loading Dataset...")
print("="*60)

train_df = pd.read_csv(f"{RAW_PATH}/train.csv")
val_df = pd.read_csv(f"{RAW_PATH}/validation.csv")
test_df = pd.read_csv(f"{RAW_PATH}/test.csv")

print("Train      :", train_df.shape)
print("Validation :", val_df.shape)
print("Test       :", test_df.shape)

# ==========================================================
# COMBINE DATASET
# ==========================================================

combined_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print("\nCombined Dataset :", combined_df.shape)

# ==========================================================
# LABEL MAPPING
# ==========================================================

emotion_map = {

    0: "Frustrated",     # sadness
    1: "Confident",      # joy
    2: "Confident",      # love
    3: "Frustrated",     # anger
    4: "Confused",       # fear
    5: "Curious"         # surprise

}

combined_df["emotion"] = combined_df["label"].map(emotion_map)

combined_df = combined_df.dropna()

combined_df = combined_df.reset_index(drop=True)

print("\nEmotion Distribution\n")
print(combined_df["emotion"].value_counts())

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z!? ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

print("\n🧹 Cleaning text...")

combined_df["clean_text"] = combined_df["text"].apply(clean_text)

# ==========================================================
# TOKENIZATION
# ==========================================================

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    combined_df["clean_text"]
)

sequences = tokenizer.texts_to_sequences(
    combined_df["clean_text"]
)

padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_SEQ_LEN,
    padding="post",
    truncating="post"
)

print("\n✅ Tokenization complete :", padded_sequences.shape)

# ==========================================================
# LABEL ENCODER
# ==========================================================

encoder = LabelEncoder()

labels = encoder.fit_transform(
    combined_df["emotion"]
)

print("\nClasses")

print(encoder.classes_)

# ==========================================================
# SPLIT BACK
# ==========================================================

train_size = len(train_df)
val_size = len(val_df)

X_train = padded_sequences[:train_size]
X_val = padded_sequences[train_size:train_size+val_size]
X_test = padded_sequences[train_size+val_size:]

y_train = labels[:train_size]
y_val = labels[train_size:train_size+val_size]
y_test = labels[train_size+val_size:]

print("\nTraining :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing :", X_test.shape)

# ==========================================================
# SAVE FILES
# ==========================================================

np.save(f"{SAVE_PATH}/X_train.npy", X_train)
np.save(f"{SAVE_PATH}/X_val.npy", X_val)
np.save(f"{SAVE_PATH}/X_test.npy", X_test)

np.save(f"{SAVE_PATH}/y_train.npy", y_train)
np.save(f"{SAVE_PATH}/y_val.npy", y_val)
np.save(f"{SAVE_PATH}/y_test.npy", y_test)

combined_df.to_csv(
    f"{SAVE_PATH}/combined_dataset.csv",
    index=False
)

with open(f"{SAVE_PATH}/tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

with open(f"{SAVE_PATH}/label_encoder.pkl","wb") as f:
    pickle.dump(encoder,f)

print("\nSaved Files")
print("----------------------------")
print("X_train.npy")
print("X_val.npy")
print("X_test.npy")
print("y_train.npy")
print("y_val.npy")
print("y_test.npy")
print("tokenizer.pkl")
print("label_encoder.pkl")
print("combined_dataset.csv")

print("\n✅ PREPROCESSING COMPLETE")