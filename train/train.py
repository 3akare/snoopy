import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import f1_score, classification_report
from model import build_model
from dataset import load_data
import tensorflow as tf

ACTIONS = ["D", "A", "V", "I", "ME", "NAME"]
NUM_SEQUENCES = 60
SEQUENCE_LENGTH = 80
DATA_PATH = os.path.join("data")
LOG_DIR = os.path.join("logs")
MODEL_SAVE_PATH = os.path.join("models", "model.h5")

x, y = load_data(ACTIONS, NUM_SEQUENCES, SEQUENCE_LENGTH, DATA_PATH)
y = to_categorical(y, num_classes=len(ACTIONS)).astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

input_shape = (SEQUENCE_LENGTH, x.shape[-1])
num_classes = len(ACTIONS)
model = build_model(input_shape, num_classes)
model.build((None,) + input_shape)
model.summary()

callbacks = [
    TensorBoard(log_dir=LOG_DIR),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
]

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=200,
    callbacks=callbacks
)

loss, acc, precision, recall = model.evaluate(test_dataset)
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")

y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print(f"F1 Score:  {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=ACTIONS))

model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
