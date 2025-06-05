from tensorflow import keras
from tensorflow.keras import layers, models

def build_bilstm_classifier(input_shape, hidden_size, num_layers, output_dim, dropout_rate):
    """
    Builds a Bidirectional LSTM classifier model using Keras Functional API.

    Args:
        input_shape (tuple): Shape of the input sequences (sequence_length, feature_dim).
        hidden_size (int): The number of units in the LSTM layer.
        num_layers (int): The number of stacked LSTM layers.
        output_dim (int): The number of output classes.
        dropout_rate (float): The dropout rate to apply.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model_input = keras.Input(shape=input_shape)
    x = model_input

    # Stack Bi-LSTM layers
    for i in range(num_layers):
        # Return sequences for all but the last LSTM layer
        return_sequences = True if i < num_layers - 1 else False
        x = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=return_sequences)
        )(x)
        # Apply dropout after each LSTM layer (except for the last one before the dense head)
        if dropout_rate > 0 and return_sequences:
            x = layers.Dropout(dropout_rate)(x)

    # Apply dropout before the final classification layer if only one layer
    if dropout_rate > 0 and num_layers == 1:
        x = layers.Dropout(dropout_rate)(x)

    # Output layer
    output = layers.Dense(output_dim, activation='softmax')(x)
    model = models.Model(inputs=model_input, outputs=output, name='BiLSTM_GestureClassifier')
    return model