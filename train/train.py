import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf
import logging
import sys

# Import custom modules
from model import build_model
from dataset import load_data

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

# Configuration constants
ACTIONS = ["D", "A","NAME"]
NUM_SEQUENCES = 60 # Number of sequences to load per action
SEQUENCE_LENGTH = 80 # Consistent sequence length
DATA_PATH = os.path.join("data")
LOG_DIR = os.path.join("logs")
MODEL_DIR = os.path.join("models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.h5") # Save best model by validation metric

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.info("Starting data loading...")
x, y = load_data(ACTIONS, NUM_SEQUENCES, SEQUENCE_LENGTH, DATA_PATH)

if x.size == 0 or y.size == 0:
    logging.critical("No data loaded. Cannot proceed with training. Exiting.")
    sys.exit(1)

logging.info(f"Loaded {len(x)} sequences with {x.shape[-1]} features per frame.")
logging.info(f"Number of classes: {len(ACTIONS)}")

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=len(ACTIONS)).astype(np.int32) # Ensure int32 for labels

# Split data
logging.info("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
logging.info(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

# Create TensorFlow datasets for optimized input pipeline
logging.info("Creating TensorFlow Datasets...")
BUFFER_SIZE = len(x_train) # Use total dataset size for shuffle buffer
BATCH_SIZE = 32 # Consistent batch size
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
logging.info("TensorFlow Datasets created.")

# Build and compile model
input_shape = (SEQUENCE_LENGTH, x.shape[-1]) # (sequence_length, feature_size)
num_classes = len(ACTIONS)
model = build_model(input_shape, num_classes)
model.build((None,) + input_shape) # Build the model to show summary and check input shape
model.summary()

# Define callbacks
logging.info("Setting up training callbacks...")
callbacks = [
    TensorBoard(log_dir=LOG_DIR),
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1), # Increased patience
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy', # Monitor validation accuracy
        mode='max', # Save model when validation accuracy is maximized
        save_best_only=True, # Only save the best model
        verbose=1
    )
]

# Train the model
logging.info("Starting model training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=200, # Max epochs
    callbacks=callbacks,
    verbose=2 # Show progress bar for each epoch
)
logging.info("Model training finished.")

# Evaluate the best model (restored by EarlyStopping)
logging.info("Evaluating model on test data...")
loss, acc, precision, recall = model.evaluate(test_dataset, verbose=0)
print(f"Final Test Accuracy:   {acc * 100:.2f}%")
print(f"Final Test Precision:  {precision * 100:.2f}%")
print(f"Final Test Recall:     {recall * 100:.2f}%")

# Generate classification report
logging.info("Generating classification report...")
y_pred = model.predict(x_test, verbose=0) # Predict on raw x_test for consistency with sklearn
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print(f"F1 Score (weighted): {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=ACTIONS))

# The best model has already been saved by ModelCheckpoint.
logging.info(f"Best model automatically saved to {MODEL_SAVE_PATH} by ModelCheckpoint callback.")