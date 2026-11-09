
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load collected data
df = pd.read_csv("data/isl_landmarks/isl_alphabet.csv")

features = [f"f{i}" for i in range(63)]

X = df[features].to_numpy(dtype=np.float32)

# Support only the letters currently collected
letters = sorted(df["label"].unique())
label_map = {letter: i for i, letter in enumerate(letters)}
y = df["label"].map(label_map).to_numpy(dtype=np.int32)

print("Letters:", letters)
print("Samples:", len(X))

# Build a small model
model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(letters), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train on all three samples (demonstration only)
model.fit(X, y, epochs=50, verbose=1)

# Save model
os.makedirs("models/trained", exist_ok=True)

model.save("models/trained/isl_test_model.h5")

with open("models/trained/isl_test_labels.json", "w") as f:
    json.dump(label_map, f)

print("Test model saved successfully!")

