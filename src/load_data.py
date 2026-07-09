import pandas as pd

train = pd.read_parquet("data/raw/train.parquet")
valid = pd.read_parquet("data/raw/validation.parquet")
test = pd.read_parquet("data/raw/test.parquet")

print("Train:", train.shape)
print("Validation:", valid.shape)
print("Test:", test.shape)

print("\nColumns:")
print(train.columns)

print("\nFirst 5 rows:")
print(train.head())