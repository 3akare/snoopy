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
from tensorflow.keras import callbacks
from bi_lstm_model import build_bilstm_classifier
from sklearn.model_selection import train_test_split
from utils import load_config, save_config, pad_or_truncate_sequence
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_CONFIG = {
    'data_dir': 'processed_data',
    'output_model_dir': '../lstm/models',
    'log_file_name': 'training_log.csv',
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'sequence_length': 80,  # Target frames per sequence (e.g., 80 frames for a 10-sec video at 8 FPS)
    'feature_dim': 126,     # (21 landmarks * 3 coords/landmark) * 2 hands = 126 features per frame
    'hidden_size': 256,
    'num_layers': 2,
    'dropout_rate': 0.3,
    'test_size': 0.2,       # Proportion of data for testing
    'validation_split': 0.2, # Proportion of training data used for validation
    'random_state': 42,
    'early_stopping_patience': 10,
    'lr_scheduler_factor': 0.5,
    'lr_scheduler_patience': 5,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bi-LSTM model for sign gesture classification using TensorFlow.")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'],
                        help=f"Directory containing processed keypoint NumPy arrays (default: {DEFAULT_CONFIG['data_dir']}).")
    parser.add_argument('--config_file', type=str, default=None,
                        help="Path to a JSON configuration file. If provided, overrides default/CLI args.")
    parser.add_argument('--save_config', action='store_true',
                        help="Save the effective configuration to a JSON file before training.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']}).")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help=f"Batch size for training (default: {DEFAULT_CONFIG['batch_size']}).")
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']}).")
    parser.add_argument('--seq_len', type=int, default=DEFAULT_CONFIG['sequence_length'],
                        help=f"Fixed sequence length for input (default: {DEFAULT_CONFIG['sequence_length']}).")
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_CONFIG['hidden_size'],
                        help=f"Hidden size of the LSTM layer (default: {DEFAULT_CONFIG['hidden_size']}).")
    parser.add_argument('--num_layers', type=int, default=DEFAULT_CONFIG['num_layers'],
                        help=f"Number of LSTM layers (default: {DEFAULT_CONFIG['num_layers']}).")
    parser.add_argument('--dropout', type=float, default=DEFAULT_CONFIG['dropout_rate'],
                        help=f"Dropout rate (default: {DEFAULT_CONFIG['dropout_rate']}).")
    parser.add_argument('--output_model_dir', type=str, default=DEFAULT_CONFIG['output_model_dir'],
                        help=f"Directory to save the trained model and confusion matrix (default: {DEFAULT_CONFIG['output_model_dir']}).")
    parser.add_argument('--log_file', type=str, default=DEFAULT_CONFIG['log_file_name'],
                        help=f"CSV file to log training metrics (default: {DEFAULT_CONFIG['log_file_name']}).")

    args = parser.parse_args()

    # Load/Update configuration
    config = DEFAULT_CONFIG.copy()
    if args.config_file:
        loaded_config = load_config(args.config_file)
        config.update(loaded_config) # Update defaults with loaded config
    
    # Override with CLI arguments if provided
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

    # Ensure output directory exists and save configuration
    os.makedirs(config['output_model_dir'], exist_ok=True)
    if args.save_config:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_config(config, os.path.join(config['output_model_dir'], f"config_{timestamp}.json"))

    # Data Loading and Preparation ---
    logging.info(f"Loading data from: {config['data_dir']}")
    all_data_paths = glob.glob(os.path.join(config['data_dir'], '**', '*.npy'), recursive=True)
    
    if not all_data_paths:
        logging.error(f"No .npy files found in '{config['data_dir']}'. "
                      "Please run extract_features.py first.")
        exit()

    all_sequences = []
    all_labels_names = []
    
    # Map labels to integers
    unique_labels = sorted(list(set([path.split(os.sep)[-2] for path in all_data_paths])))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    num_classes = len(unique_labels)

    label_map_path = os.path.join(config['output_model_dir'], 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(idx_to_label, f, indent=4)
    logging.info(f"Label mapping saved to {label_map_path}")

    for path in all_data_paths:
        try:
            keypoints = np.load(path)
            # Ensure the feature dimension matches expected
            if keypoints.shape[1] != config['feature_dim']:
                logging.warning(f"Feature dimension mismatch for {path}. Expected {config['feature_dim']}, got {keypoints.shape[1]}. Skipping.")
                continue

            processed_sequence = pad_or_truncate_sequence(keypoints, config['sequence_length'])
            all_sequences.append(processed_sequence)

            label = path.split(os.sep)[-2]
            all_labels_names.append(label)
        except Exception as e:
            logging.error(f"Error loading or processing {path}: {e}. Skipping.")
            continue

    if not all_sequences:
        logging.error("No valid sequences found after loading and preprocessing. Exiting.")
        exit()

    X = np.array(all_sequences, dtype=np.float32)
    y_labels = np.array([label_to_idx[name] for name in all_labels_names], dtype=np.int32)

    # Split data into training and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_labels, test_size=config['test_size'],
        random_state=config['random_state'], stratify=y_labels
    )
    
    logging.info(f"Total samples: {len(all_data_paths)}")
    logging.info(f"Training + Validation samples: {X_train_val.shape[0]}")
    logging.info(f"Testing samples: {X_test.shape[0]}")
    logging.info(f"Number of classes: {num_classes}")

    # Model Initialization
    input_shape = (config['sequence_length'], config['feature_dim'])
    model = build_bilstm_classifier(
        input_shape=input_shape,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_dim=num_classes,
        dropout_rate=config['dropout_rate']
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy', # Use for integer labels
        metrics=['accuracy']
    )

    model.summary(print_fn=logging.info) # Direct model summary to logging

    # Callbacks
    log_csv_path = os.path.join(config['output_model_dir'], config['log_file_name'])
    csv_logger = callbacks.CSVLogger(log_csv_path, append=True)

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'], min_lr=1e-7, verbose=1
    )

    model_checkpoint_path = os.path.join(config['output_model_dir'], 'best_model_tf.keras')
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Training
    logging.info("\n--- Starting Training ---")
    history = model.fit(
        X_train_val, y_train_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        callbacks=[csv_logger, early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    logging.info("--- Training Complete ---")

    # Evaluation
    logging.info("\n--- Starting Evaluation ---")
    # Load the best model weights if saved by ModelCheckpoint
    if os.path.exists(model_checkpoint_path):
        logging.info(f"Best model loaded for final evaluation from {model_checkpoint_path}.")
    else:
        logging.warning("Best model not found at checkpoint path. Evaluating with the last trained model state.")

    # Evaluate on the test set
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=config['batch_size'], verbose=0)
    logging.info(f"Overall Test Loss: {loss:.4f}")
    logging.info(f"Overall Test Accuracy: {accuracy:.4f}")

    # Predict on test set for precision, recall, F1-score, and confusion matrix
    y_pred_probs = model.predict(X_test, batch_size=config['batch_size'])
    y_pred = np.argmax(y_pred_probs, axis=1)

    overall_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    logging.info(f"Overall Test Precision: {overall_precision:.4f}")
    logging.info(f"Overall Test Recall: {overall_recall:.4f}")
    logging.info(f"Overall Test F1-Score: {overall_f1:.4f}")

    # Plot Training History
    logging.info("\n--- Plotting Training History ---")
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    history_plot_path = os.path.join(config['output_model_dir'], 'training_history.png')
    plt.tight_layout()
    plt.savefig(history_plot_path)
    logging.info(f"Training history plot saved to {history_plot_path}")

    # Generate Confusion Matrix
    logging.info("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_label[i] for i in sorted(idx_to_label.keys())],
                yticklabels=[idx_to_label[i] for i in sorted(idx_to_label.keys())])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_save_path = os.path.join(config['output_model_dir'], 'confusion_matrix.png')
    plt.savefig(cm_save_path)
    logging.info(f"Confusion matrix saved to {cm_save_path}")
    logging.info("--- Evaluation Complete ---")