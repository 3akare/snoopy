import os
import json
import glob
import logging
import argparse
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
from bi_lstm_model import build_ctc_model
from sklearn.model_selection import train_test_split
from utils import load_config, save_config, pad_or_truncate_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_CONFIG = {
    'data_dir': 'processed_data',
    'output_model_dir': '../lstm/models',
    'log_file_name': 'training_log.csv',
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'sequence_length': 80,
    'feature_dim': 126,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout_rate': 0.3,
    'test_size': 0.2,
    'validation_split': 0.2, # This will be used to split the train_val set
    'random_state': 42,
    'early_stopping_patience': 10,
    'lr_scheduler_factor': 0.5,
    'lr_scheduler_patience': 5,
}

# Define the character set for CTC model. This must include every character your model can predict.
CHARACTERS = sorted(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '])
char_to_num = layers.StringLookup(vocabulary=list(CHARACTERS), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def ctc_loss(y_true, y_pred):
    """The CTC loss function."""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def add_noise(sequence: np.ndarray, noise_level=0.001) -> np.ndarray:
    """Adds Gaussian noise to a keypoint sequence."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_sequence(sequence: np.ndarray, scale_range=(0.8, 1.2)) -> np.ndarray:
    """Scales a keypoint sequence."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return sequence * scale_factor

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """Applies a random set of augmentations to a sequence."""
    augmented_sequence = sequence.copy()
    if np.random.rand() < 0.5:
        augmented_sequence = add_noise(augmented_sequence)
    if np.random.rand() < 0.5:
        augmented_sequence = scale_sequence(augmented_sequence)
    return augmented_sequence

def data_generator(X_data, y_data, batch_size, augment=False):
    """Generates batches of data with optional on-the-fly augmentation."""
    num_samples = len(X_data)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_data[batch_indices]
            y_batch = y_data[batch_indices]

            if augment:
                X_batch_augmented = np.array([augment_sequence(seq) for seq in X_batch])
                yield X_batch_augmented, y_batch
            else:
                yield X_batch, y_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bi-LSTM CTC model for sign gesture classification with data augmentation.")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'])
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--save_config', action='store_true')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--seq_len', type=int, default=DEFAULT_CONFIG['sequence_length'])
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_CONFIG['hidden_size'])
    parser.add_argument('--num_layers', type=int, default=DEFAULT_CONFIG['num_layers'])
    parser.add_argument('--dropout', type=float, default=DEFAULT_CONFIG['dropout_rate'])
    parser.add_argument('--output_model_dir', type=str, default=DEFAULT_CONFIG['output_model_dir'])
    parser.add_argument('--log_file', type=str, default=DEFAULT_CONFIG['log_file_name'])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config_file:
        config.update(load_config(args.config_file))
    
    config['data_dir'] = args.data_dir
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['sequence_length'] = args.seq_len
    config['hidden_size'] = args.hidden_size
    config['num_layers'] = args.num_layers
    config['dropout_rate'] = args.dropout
    config['output_model_dir'] = args.output_model_dir
    config['log_file_name'] = args.log_file

    os.makedirs(config['output_model_dir'], exist_ok=True)
    if args.save_config:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_config(config, os.path.join(config['output_model_dir'], f"config_{timestamp}.json"))

    logging.info(f"Loading data from: {config['data_dir']}")
    all_data_paths = glob.glob(os.path.join(config['data_dir'], '**', '*.npy'), recursive=True)
    
    if not all_data_paths:
        logging.error(f"No .npy files found in '{config['data_dir']}'.")
        exit()

    vocab_path = os.path.join(config['output_model_dir'], 'char_vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(char_to_num.get_vocabulary(), f, indent=4)
    logging.info(f"Character vocabulary saved to {vocab_path}")

    all_sequences = []
    all_text_labels = []

    for path in all_data_paths:
        try:
            keypoints = np.load(path)
            if keypoints.shape[1] != config['feature_dim']:
                logging.warning(f"Feature dimension mismatch for {path}. Skipping.")
                continue
            processed_sequence = pad_or_truncate_sequence(keypoints, config['sequence_length'])
            all_sequences.append(processed_sequence)
            label_text = path.split(os.sep)[-2]
            all_text_labels.append(label_text)
        except Exception as e:
            logging.error(f"Error loading or processing {path}: {e}. Skipping.")
            continue

    if not all_sequences:
        logging.error("No valid sequences found. Exiting.")
        exit()

    X = np.array(all_sequences, dtype=np.float32)
    y_labels_as_chars = [tf.strings.unicode_split(label, 'UTF-8') for label in all_text_labels]
    y_labels_as_nums = [char_to_num(chars) for chars in y_labels_as_chars]
    y = tf.keras.preprocessing.sequence.pad_sequences(y_labels_as_nums, padding='post')

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # Split the training+validation set into a final training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=config['validation_split'], 
        random_state=config['random_state']
    )

    logging.info(f"Total samples: {len(all_data_paths)}")
    logging.info(f"Training samples: {X_train.shape[0]}")
    logging.info(f"Validation samples: {X_val.shape[0]}")
    logging.info(f"Testing samples: {X_test.shape[0]}")

    input_shape = (config['sequence_length'], config['feature_dim'])
    num_classes = len(char_to_num.get_vocabulary())
    
    model = build_ctc_model(
        input_shape=input_shape,
        num_layers=config['num_layers'],
        hidden_size=config['hidden_size'],
        dropout_rate=config['dropout_rate'],
        num_classes=num_classes
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=ctc_loss
    )

    model.summary(print_fn=logging.info)

    log_csv_path = os.path.join(config['output_model_dir'], config['log_file_name'])
    csv_logger = callbacks.CSVLogger(log_csv_path, append=True)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['lr_scheduler_factor'], patience=config['lr_scheduler_patience'], min_lr=1e-7, verbose=1)
    model_checkpoint_path = os.path.join(config['output_model_dir'], 'best_model_tf.keras')
    model_checkpoint = callbacks.ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    train_gen = data_generator(X_train, y_train, config['batch_size'], augment=True)
    val_gen = data_generator(X_val, y_val, config['batch_size'], augment=False)

    logging.info("\n--- Starting Training with Data Augmentation ---")
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, len(X_train) // config['batch_size']),
        validation_data=val_gen,
        validation_steps=max(1, len(X_val) // config['batch_size']),
        epochs=config['epochs'],
        callbacks=[csv_logger, early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    logging.info("--- Training Complete ---")

    logging.info("\n--- Starting Evaluation ---")
    if os.path.exists(model_checkpoint_path):
        logging.info(f"Best model loaded for final evaluation from {model_checkpoint_path}.")
    else:
        logging.warning("Best model not found. Evaluating with the last trained model state.")

    test_loss = model.evaluate(X_test, y_test, batch_size=config['batch_size'], verbose=0)
    logging.info(f"Overall Test Loss (CTC): {test_loss:.4f}")

    logging.info("\n--- Plotting Training History ---")
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model CTC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    history_plot_path = os.path.join(config['output_model_dir'], 'training_history_loss.png')
    plt.tight_layout()
    plt.savefig(history_plot_path)
    logging.info(f"Training history plot saved to {history_plot_path}")
    logging.info("--- Process Complete ---")
    