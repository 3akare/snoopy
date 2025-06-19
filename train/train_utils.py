import json
import logging
import numpy as np

def save_config(config, filepath):
    """Saves the configuration to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Configuration saved to {filepath}")

def load_config(filepath):
    """Loads configuration from a JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    logging.info(f"Configuration loaded from {filepath}")
    return config

def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int, padding_value: float = 0.0) -> np.ndarray:
    """
    Pads or truncates a NumPy sequence to a target length.
    """
    if sequence.shape[0] > target_length:
        return sequence[:target_length, :]
    elif sequence.shape[0] < target_length:
        padding = np.full((target_length - sequence.shape[0], sequence.shape[1]), padding_value, dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)
    return sequence