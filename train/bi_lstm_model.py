from tensorflow import keras
from tensorflow.keras import layers, models

def build_ctc_model(input_shape, num_layers, hidden_size, dropout_rate, num_classes):
    """
    Builds a Bidirectional LSTM model designed for CTC loss.

    Args:
        input_shape (tuple): Shape of input sequences (sequence_length, feature_dim).
        num_layers (int): Number of stacked LSTM layers.
        hidden_size (int): Number of units in the LSTM layer.
        dropout_rate (float): Dropout rate.
        num_classes (int): Number of output classes (e.g., number of characters in the alphabet).

    Returns:
        tf.keras.Model: The Keras model ready for CTC training.
    """
    model_input = keras.Input(shape=input_shape, name="input")
    x = model_input

    # Stack Bi-LSTM layers
    for i in range(num_layers):
        # All layers must return sequences for CTC
        x = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True, dropout=dropout_rate)
        )(x)

    # Output layer for CTC: Dense layer with softmax activation.
    # The number of units is num_classes + 1 to account for the CTC blank token.
    output = layers.Dense(num_classes + 1, activation='softmax', name='output')(x)

    # Create the model
    model = models.Model(inputs=model_input, outputs=output, name='CTC_GestureClassifier')
    return model