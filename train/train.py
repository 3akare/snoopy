import os
import sys
import random
import logging
import numpy as np
import tensorflow as tf
from dataset import load_data
from model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED) 


ACTIONS = []
NUM_SEQUENCES = 60
SEQUENCE_LENGTH = 80
DATA_PATH = os.path.join("data")
LOG_DIR = os.path.join("logs")
MODEL_DIR = os.path.join("model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model.keras")

# label_to_index = {action: i for i, action in enumerate(ACTIONS)}

# Define weights based on observed poor performance. Higher weight for worse recall.
# You can start with something like this:
# class_weights_dict = {
#     label_to_index["D"]: 2.0,    # Recall was ~0.50
#     label_to_index["A"]: 15.0,   # Recall was 0.00
#     label_to_index["V"]: 10.0,   # Recall was ~0.08
#     label_to_index["I"]: 15.0,   # Recall was 0.00
#     label_to_index["S"]: 15.0,   # Recall was 0.00
#     label_to_index["T"]: 15.0,   # Recall was 0.00
#     label_to_index["E"]: 1.0,    # Recall was ~0.83
#     label_to_index["ME"]: 1.0,   # Recall was ~0.83
#     label_to_index["NAME"]: 1.0  # Recall was 1.00
# }
# logging.info(f"Using class weights: {class_weights_dict}")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.info("Starting data loading...")
x, y = load_data(ACTIONS, NUM_SEQUENCES, SEQUENCE_LENGTH, DATA_PATH)

if x.size == 0 or y.size == 0:
    logging.critical("No data loaded. Cannot proceed with training. Exiting.")
    sys.exit(1)

logging.info(f"Loaded {len(x)} sequences with {x.shape[-1]} features per frame.")
logging.info(f"Number of classes: {len(ACTIONS)}")
y = to_categorical(y, num_classes=len(ACTIONS)).astype(np.int32)

logging.info("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, stratify=y, random_state=SEED)
logging.info(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

# Create TensorFlow datasets for optimized input pipeline
logging.info("Creating TensorFlow Datasets...")
BUFFER_SIZE = len(x_train)
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
logging.info("TensorFlow Datasets created.")

# Build and compile model
input_shape = (SEQUENCE_LENGTH, x.shape[-1])
num_classes = len(ACTIONS)
model = build_model(input_shape, num_classes)
model.build((None,) + input_shape)
model.summary()

# Define callbacks
logging.info("Setting up training callbacks...")
callbacks = [
    TensorBoard(log_dir=LOG_DIR),
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
logging.info("Starting model training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=200,
    callbacks=callbacks,
    verbose=2,
    # class_weight=class_weights_dict
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
y_pred = model.predict(x_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print(f"F1 Score (weighted): {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=ACTIONS))

logging.info(f"Best model saved to {MODEL_SAVE_PATH} by ModelCheckpoint callback.")