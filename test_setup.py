import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import transformers
import torch

print("=" * 50)
print("TensorFlow :", tf.__version__)
print("NumPy      :", np.__version__)
print("Pandas     :", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Transformers:", transformers.__version__)
print("PyTorch     :", torch.__version__)

print("\nGPU Count:", len(tf.config.list_physical_devices("GPU")))
print("=" * 50)
print("Environment setup successful!")