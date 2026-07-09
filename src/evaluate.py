# ==========================================================
# SMARTBRIDGE AI LEARNING ASSISTANT
# TASK 3 : MODEL EVALUATION
# ==========================================================

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

try:
    # Prefer tensorflow.keras when available (commonly used in TF installs)
    from tensorflow.keras.models import load_model
except (ImportError, ModuleNotFoundError):
    # Fallback to standalone keras if tensorflow.keras is not present
    from keras.models import load_model

from model import focal_loss

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

model = load_model(

    "models/bilstm_emotion_model.keras",

    custom_objects={
        "loss": focal_loss()
    }

)

print("✅ Model Loaded Successfully")

# ==========================================================
# LOAD TEST DATA
# ==========================================================

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("\nTest Data Shape :", X_test.shape)

# ==========================================================
# PREDICTION
# ==========================================================

print("\nPredicting...")

y_prob = model.predict(X_test, verbose=1)

y_pred = np.argmax(y_prob, axis=1)

# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"TEST ACCURACY : {accuracy:.4f}")
print("=" * 60)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nCLASSIFICATION REPORT\n")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=label_encoder.classes_,

        digits=4

    )

)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=label_encoder.classes_,

    yticklabels=label_encoder.classes_

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("models/confusion_matrix.png")

plt.show()

print("\n✅ Confusion Matrix Saved")

print("models/confusion_matrix.png")

# ==========================================================
# PREDICTION DISTRIBUTION
# ==========================================================

print("\nPrediction Distribution\n")

unique, counts = np.unique(y_pred, return_counts=True)

for label, count in zip(unique, counts):

    print(f"{label_encoder.classes_[label]} : {count}")

print("\n✅ Evaluation Completed Successfully")