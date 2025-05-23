import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam # Explicitly import Adam

def build_model(input_shape, num_classes):
    """
    Builds a Bidirectional LSTM model for gesture recognition.

    Args:
        input_shape (tuple): Shape of the input sequences (e.g., (sequence_length, feature_size)).
        num_classes (int): Number of distinct action classes.

    Returns:
        tf.keras.Model: Compiled TensorFlow Keras model.
    """
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')), # Increased units for potentially more capacity, added activation
        Dropout(0.4), # Increased dropout slightly
        Bidirectional(LSTM(64, activation='relu')), # Increased units, added activation
        Dropout(0.4), # Increased dropout slightly
        Dense(64, activation='relu'), # Increased units
        Dropout(0.4), # Increased dropout slightly
        Dense(num_classes, activation='softmax')
    ])
    
    # Use Adam optimizer with a learning rate, which can be tuned
    optimizer = Adam(learning_rate=0.0005) # Starting with a slightly lower learning rate

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            CategoricalAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    return model