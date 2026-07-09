# ==========================================================
# SMARTBRIDGE AI LEARNING ASSISTANT
# TASK 3 : BiLSTM MODEL TRAINING
# ==========================================================

import os
import time
import numpy as np
import tensorflow as tf

from model import build_model

# ==========================================================
# CREATE MODELS DIRECTORY
# ==========================================================

os.makedirs("models", exist_ok=True)

# ==========================================================
# LOAD PREPROCESSED DATA
# ==========================================================

print("=" * 60)
print("Loading Preprocessed Data...")
print("=" * 60)

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")

y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

print(f"Training Data   : {X_train.shape}")
print(f"Validation Data : {X_val.shape}")

# ==========================================================
# BUILD MODEL
# ==========================================================

model = build_model()

print("\n")
print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    verbose=1

)

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath="models/bilstm_emotion_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

callbacks = [

    early_stop,

    reduce_lr,

    checkpoint

]

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

start_time = time.time()

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_val, y_val),

    epochs=10,

    batch_size=128,

    callbacks=callbacks,

    verbose=1

)

training_time = (time.time() - start_time) / 60

print("\n")
print("=" * 60)
print(f"Training Completed in {training_time:.2f} minutes")
print("=" * 60)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

model.save("models/bilstm_emotion_model.keras")

print("\n✅ Model Saved Successfully")

print("Location : models/bilstm_emotion_model.keras")

# ==========================================================
# FINAL METRICS
# ==========================================================

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

train_loss = history.history["loss"][-1]
val_loss = history.history["val_loss"][-1]

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Training Accuracy   : {train_acc:.4f}")
print(f"Validation Accuracy : {val_acc:.4f}")

print(f"Training Loss       : {train_loss:.4f}")
print(f"Validation Loss     : {val_loss:.4f}")

print("=" * 60)