import os
import numpy as np
import logging
import sys

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

def load_data(actions, no_sequences, sequence_length, data_path):
    """
    Loads pre-processed keypoint sequences and their corresponding labels from disk.

    Args:
        actions (list): List of action names (e.g., ["D", "A", "V"]).
        no_sequences (int): Expected number of sequences per action (or max to check).
        sequence_length (int): The fixed length of each sequence.
        data_path (str): Base path to the data directory.

    Returns:
        tuple: (sequences (numpy array), labels (numpy array))
    """
    sequences = []
    labels = []
    label_map = {action: idx for idx, action in enumerate(actions)}
    
    logging.info(f"Loading data from {data_path} for actions: {actions}...")
    
    for action in actions:
        action_dir = os.path.join(data_path, action)
        if not os.path.isdir(action_dir):
            logging.warning(f"Action directory not found: {action_dir}. Skipping.")
            continue
            
        # Iterate up to `no_sequences` or through all available directories
        # It's safer to iterate through actual directories found
        found_sequences_for_action = 0
        for sequence_num_str in sorted(os.listdir(action_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):
            if not sequence_num_str.isdigit():
                continue # Skip non-numeric directories
            
            sequence_dir = os.path.join(action_dir, sequence_num_str)
            npy_path = os.path.join(sequence_dir, "sequence_data.npy")
            
            if not os.path.exists(npy_path):
                logging.debug(f"Sequence data file not found: {npy_path}. Skipping.")
                continue
            
            try:
                sequence_data = np.load(npy_path)
                
                # Validate sequence shape: (SEQUENCE_LENGTH, feature_size)
                # Feature size is determined by the output of extract_keypoints (e.g., 126)
                expected_feature_size = 126 # Assuming 21*3 (LH) + 21*3 (RH)
                
                if sequence_data.shape != (sequence_length, expected_feature_size):
                    logging.warning(f"Incorrect shape for sequence {npy_path}: expected ({sequence_length}, {expected_feature_size}), got {sequence_data.shape}. Skipping.")
                    continue
                
                sequences.append(sequence_data)
                labels.append(label_map[action])
                found_sequences_for_action += 1

                if found_sequences_for_action >= no_sequences:
                    logging.info(f"Loaded {no_sequences} sequences for action '{action}'. Stopping for this action.")
                    break # Stop if we've loaded enough sequences for this action

            except Exception as e:
                logging.error(f"Error loading or validating sequence {npy_path}: {e}", exc_info=True)
                continue
    
    if not sequences:
        logging.critical("No sequences loaded. Check data_path, action names, and sequence directories.")
        return np.array([]), np.array([]) # Return empty arrays if no data
        
    sequences_np = np.array(sequences, dtype=np.float32) # Ensure float32
    labels_np = np.array(labels, dtype=np.int32) # Labels are integers
    
    logging.info(f"Successfully loaded {len(sequences_np)} total sequences.")
    return sequences_np, labels_np