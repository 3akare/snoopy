import os
import numpy as np

def load_data(actions, no_sequences, sequence_length, data_path):
    sequences, labels = [], []
    label_map = {action: num for num, action in enumerate(actions)}
    for action in actions:
        for sequence in range(no_sequences):
            npy_path = os.path.join(data_path, action, str(sequence), "sequence_data.npy")
            if not os.path.exists(npy_path):
                print(f"Warning: Sequence file not found for {action}/Sequence {sequence}. Skipping.")
                continue
            try:
                sequence_data = np.load(npy_path)
                if sequence_data.shape[0] != sequence_length:
                     print(f"Warning: Sequence {action}/Sequence {sequence} has unexpected length {sequence_data.shape[0]}. Expected {sequence_length}. Skipping or padding/truncating might be needed.")
                     continue
                sequences.append(sequence_data)
                labels.append(label_map[action])

            except Exception as e:
                 print(f"Error loading sequence file {npy_path}: {e}. Skipping.")
                 continue
    return np.array(sequences), np.array(labels)