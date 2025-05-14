import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

def build_model(input_shape, num_classes):
  model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),  
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy', 
      metrics=[
          CategoricalAccuracy(name='accuracy'),
          Precision(name='precision'),
          Recall(name='recall')
      ]
  )
  return model
