import os
import numpy as np

def load_data(actions, no_sequences, sequence_length, data_path):
    sequences, labels = []
    label_map = {action: idx for idx, action in enumerate(actions)}
    
    for action in actions:
        for sequence in range(no_sequences):
            npy_path = os.path.join(data_path, action, str(sequence), "sequence_data.npy")
            if not os.path.exists(npy_path):
                continue
            sequence_data = np.load(npy_path)
            if sequence_data.shape[0] != sequence_length:
                continue
            sequences.append(sequence_data)
            labels.append(label_map[action])

    sequences = np.array(sequences)
    labels = np.array(labels)

    # Shuffle data before returning
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    return sequences[indices], labels[indices]
