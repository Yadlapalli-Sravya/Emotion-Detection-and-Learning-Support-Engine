from datasets import load_dataset
import pandas as pd
import os

# Create folder
os.makedirs("data/raw", exist_ok=True)

print("Downloading Emotion Dataset...")

dataset = load_dataset("dair-ai/emotion")

# Save each split
dataset["train"].to_pandas().to_csv(
    "data/raw/train.csv",
    index=False
)

dataset["validation"].to_pandas().to_csv(
    "data/raw/validation.csv",
    index=False
)

dataset["test"].to_pandas().to_csv(
    "data/raw/test.csv",
    index=False
)

print("✅ Dataset Downloaded Successfully!")