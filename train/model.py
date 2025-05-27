import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
        Dropout(0.4),
        Bidirectional(LSTM(64, activation='relu')),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0005)
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