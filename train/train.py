import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import f1_score, classification_report
from model import build_model
from dataset import load_data
from augmentation import apply_augmentations
import tensorflow as tf

ACTIONS = ["D", "A", "V", "I", "D", "ME", "NAME"]
NUM_SEQUENCES = 60
SEQUENCE_LENGTH = 80
DATA_PATH = os.path.join("data")
LOG_DIR = os.path.join("logs")
MODEL_SAVE_PATH = os.path.join("models", "model.h5")

print("Loading data...")
x, y = load_data(ACTIONS, NUM_SEQUENCES, SEQUENCE_LENGTH, DATA_PATH)
y = to_categorical(y, num_classes=len(ACTIONS)).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# check distribution
train_labels = np.argmax(y_train, axis=1)
test_labels = np.argmax(y_test, axis=1)
unique_train, counts_train = np.unique(train_labels, return_counts=True)
unique_test, counts_test = np.unique(test_labels, return_counts=True)
print("\nTraining set class distribution:", dict(zip(unique_train, counts_train)))
print("Test set class distribution:", dict(zip(unique_test, counts_test)))

# Data augementation
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=len(x_train))
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (SEQUENCE_LENGTH, x.shape[-1])
num_classes = len(ACTIONS)
model = build_model(input_shape, num_classes)
model.build((None,) + input_shape)
model.summary()

tb_callback = TensorBoard(log_dir=LOG_DIR)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("Training model...")
model.fit(train_dataset,
          epochs=200,
          validation_data=test_dataset,
          callbacks=[tb_callback, early_stopping])

print("Evaluating model...")
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