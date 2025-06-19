from tensorflow import keras
from tensorflow.keras import layers, models

def build_ctc_model(input_shape, num_layers, hidden_size, dropout_rate, num_classes):
    """
    Builds a Bidirectional LSTM model designed for CTC loss.
    """
    model_input = keras.Input(shape=input_shape, name="input")
    x = model_input

    for i in range(num_layers):
        x = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True, dropout=dropout_rate)
        )(x)

    # Output layer for CTC requires a probability distribution over characters for each time step.
    # The number of units is num_classes + 1 to account for the CTC blank token.
    output = layers.Dense(num_classes + 1, activation='softmax', name='output')(x)

    model = models.Model(inputs=model_input, outputs=output, name='CTC_GestureClassifier')
    return model