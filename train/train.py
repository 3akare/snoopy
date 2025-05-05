import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import f1_score, classification_report
from model import build_model
from dataset import load_data

# Training parameters
ACTIONS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    "Name",
    "Learn",
    "Restroom",
    "No",
    "What",
    "Sign",
    "Where",
    "Sister",
    "Nice",
    "Not",
    "Classroom",
    "Girl-friend",
    "You",
    "Student",
    "Buy",
    "Brother",
    "Meet",
    "Teacher",
    "Food",
    "Have"
]
no_sequences = 30
sequence_length = 30
DATA_PATH = os.path.join("data")
LOG_DIR = os.path.join("logs")
MODEL_SAVE_PATH = os.path.join("models", "model.h5")

print("Loading data...")
x, y = load_data(ACTIONS, no_sequences, sequence_length, DATA_PATH)
y = to_categorical(y).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

input_shape = (sequence_length, x.shape[-1])
num_classes = len(ACTIONS)
model = build_model(input_shape, num_classes)
model.build((None,) + input_shape)
model.summary()

tb_callback = TensorBoard(log_dir=LOG_DIR)

print("Training model...")
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[tb_callback])

print("Evaluating model...")
loss, acc, precision, recall = model.evaluate(x_test, y_test)
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print(f"F1 Score:  {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=actions))
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
