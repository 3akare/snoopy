# model/train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from model import build_model
from dataset import load_data

# Training parameters
actions = ["hello", "thanks", "i love you"]
no_sequences = 30
sequence_length = 30
DATA_PATH = os.path.join("data", "MP_Data")
LOG_DIR = os.path.join("logs")
MODEL_SAVE_PATH = os.path.join("models", "model.h5")

print("Loading data...")
x, y = load_data(actions, no_sequences, sequence_length, DATA_PATH)
y = to_categorical(y).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

input_shape = (sequence_length, x.shape[-1])  # e.g. (30, 1662)
num_classes = len(actions)
model = build_model(input_shape, num_classes)
model.summary()

tb_callback = TensorBoard(log_dir=LOG_DIR)

print("Training model...")
model.fit(x_train, y_train, epochs=100, callbacks=[tb_callback])

loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

