# model/dataset.py
import os
import numpy as np

def load_data(actions, no_sequences, sequence_length, data_path):
    """
    Load gesture keypoint data.
    
    Args:
      actions (list): List of action names.
      no_sequences (int): Number of sequences per action.
      sequence_length (int): Number of frames per sequence.
      data_path (str): Path to the dataset directory.
      
    Returns:
      Tuple (x, y) where x is the data array and y are labels.
    """
    sequences, labels = [], []
    label_map = {action: num for num, action in enumerate(actions)}
    
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                npy_path = os.path.join(data_path, action, str(sequence), f"{frame_num}.npy")
                res = np.load(npy_path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    
    return np.array(sequences), np.array(labels)

