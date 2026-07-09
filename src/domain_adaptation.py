# ==========================================================
# SMARTBRIDGE AI LEARNING ASSISTANT
# TASK 4 : DOMAIN ADAPTIVE FINE-TUNING
# ==========================================================

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from model import focal_loss

# ==========================================================
# CREATE MODELS DIRECTORY
# ==========================================================

os.makedirs("models", exist_ok=True)

# ==========================================================
# LOAD PREPROCESSED DATA
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")

y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

print("Training Shape :", X_train.shape)
print("Validation Shape :", X_val.shape)

# ==========================================================
# LOAD BASELINE MODEL
# ==========================================================

print("\nLoading BiLSTM Model...")

baseline_model = load_model(

    "models/bilstm_emotion_model.keras",

    custom_objects={
        "loss": focal_loss()
    }

)

print("✅ Baseline Model Loaded")

# ==========================================================
# FREEZE EMBEDDING LAYER
# ==========================================================

baseline_model.layers[0].trainable = False

print("✅ Embedding Layer Frozen")

# ==========================================================
# RECOMPILE
# ==========================================================

baseline_model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4

    ),

    loss=focal_loss(),

    metrics=["accuracy"]

)

print("✅ Adaptive Model Ready")

# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True

    )

]

# ==========================================================
# FINE-TUNING
# ==========================================================

print("\n🚀 Fine-tuning on student domain...\n")

start = time.time()

history = baseline_model.fit(

    X_train,

    y_train,

    validation_data=(X_val, y_val),

    epochs=8,

    batch_size=64,

    callbacks=callbacks,

    verbose=1

)

minutes = (time.time() - start) / 60

print(f"\nTraining Time : {minutes:.2f} minutes")

# ==========================================================
# SAVE MODEL
# ==========================================================

baseline_model.save(

    "models/bilstm_student_adaptive.keras"

)

print("\n✅ Adaptive Model Saved")

print("models/bilstm_student_adaptive.keras")

# ==========================================================
# PLOT LOSS
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history["loss"],

    label="Train"

)

plt.plot(

    history.history["val_loss"],

    label="Validation"

)

plt.title("Domain-Adaptive Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "models/domain_adaptive_loss.png"

)

plt.show()

print("\n✅ Loss Graph Saved")

print("models/domain_adaptive_loss.png")