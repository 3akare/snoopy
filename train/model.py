# model/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(input_shape, num_classes):
    """
    Build and compile the LSTM model.
    
    Args:
      input_shape (tuple): (sequence_length, feature_size)
      num_classes (int): Number of action classes.
    
    Returns:
      Compiled Keras model.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model

