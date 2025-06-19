import os
import sys
import json
import grpc
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import prediction_services_pb2
import prediction_services_pb2_grpc
from concurrent import futures
from utils import pad_or_truncate_sequence

# Configuration
MAX_MESSAGE_LENGTH = 1024 * 1024 * 50
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, 'models', 'best_model_tf.keras')
VOCAB_PATH = os.path.join(MODEL_DIR, 'models', 'char_vocab.json')
SEQUENCE_LENGTH = 80
FEATURE_SIZE = 126
WINDOW_STRIDE = 5

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Character Vocabulary
try:
    with open(VOCAB_PATH, 'r') as f:
        VOCAB = json.load(f)
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=VOCAB, mask_token=None, invert=True
    )
    logging.info(f"Successfully loaded character vocabulary.")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load character vocabulary. Exiting. Error: {e}", exc_info=True)
    sys.exit(1)

# Load the TensorFlow Model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False) # Set compile=False for CTC
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
    model.predict(dummy_input)
    logging.info(f"Successfully loaded Keras model from {MODEL_PATH}")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load Keras model. Exiting. Error: {e}", exc_info=True)
    sys.exit(1)

def ctc_decode_predictions(pred):
    """Decodes the raw output of the CTC model."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For better results, beam search can be used.
    results = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    
    output_text = []
    for res in results:
        # Convert numeric results back to text characters
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

class LstmPredictionService(prediction_services_pb2_grpc.LstmServiceServicer):
    def Predict(self, request, context):
        """
        Predicts sign language gestures from a sequence of keypoints using a CTC-based model.
        """
        try:
            if not request.gestures:
                context.set_details("No gestures provided in the request.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return prediction_services_pb2.LstmResponse(translated_text="")

            input_frames = [frame.keypoints for frame in request.gestures[0].frames]
            live_sequence_np = np.array(input_frames, dtype=np.float32)

            if live_sequence_np.ndim != 2 or live_sequence_np.shape[1] != FEATURE_SIZE:
                context.set_details(f"Invalid keypoint dimension. Expected (N, {FEATURE_SIZE})")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return prediction_services_pb2.LstmResponse(translated_text="Error: Invalid data format.")

            total_frames = live_sequence_np.shape[0]
            predicted_texts = []

            logging.info(f"Received {total_frames} frames. Applying sliding window with SEQUENCE_LENGTH={SEQUENCE_LENGTH}, WINDOW_STRIDE={WINDOW_STRIDE}.")
            for i in range(0, total_frames - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
                window_sequence = live_sequence_np[i : i + SEQUENCE_LENGTH, :]
                sequence_to_predict = pad_or_truncate_sequence(window_sequence, SEQUENCE_LENGTH, padding_value=0.0)
                sequence_to_predict = np.expand_dims(sequence_to_predict, axis=0)

                prediction = model.predict(sequence_to_predict, verbose=0)
                
                decoded_prediction = ctc_decode_predictions(prediction)
                predicted_texts.extend(decoded_prediction)
            
            # Post-process the list of predicted text chunks to form a coherent sentence.
            # This simple approach joins unique, non-empty predictions.
            if predicted_texts:
                response_text = " ".join(dict.fromkeys(filter(None, predicted_texts)))
            else:
                response_text = ""
            
            logging.info(f"Finished processing. Result: '{response_text}'")
            return prediction_services_pb2.LstmResponse(translated_text=response_text)

        except Exception as e:
            logging.error(f"Error in Predict RPC: {e}", exc_info=True)
            context.set_details(f"Internal server error: {type(e).__name__}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return prediction_services_pb2.LstmResponse(translated_text="Error processing prediction.")


def serve():
    """Starts the gRPC server."""
    max_workers = os.cpu_count() or 4
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    prediction_services_pb2_grpc.add_LstmServiceServicer_to_server(LstmPredictionService(), server)
    server.add_insecure_port("[::]:50051")
    logging.info(f"LSTM gRPC Server started on port 50051 with CTC-based prediction.")

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server shutting down.")
        server.stop(0)
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    serve()