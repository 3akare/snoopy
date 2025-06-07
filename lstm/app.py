import os
import sys
import json
import grpc
import logging
import numpy as np
import tensorflow as tf
import prediction_services_pb2
import prediction_services_pb2_grpc
from concurrent import futures


# Configuration
MAX_MESSAGE_LENGTH = 1024 * 1024 * 50  # 50 MB
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, 'models', 'best_model_tf.keras')
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'models', 'label_map.json')
SEQUENCE_LENGTH = 80
FEATURE_SIZE = 126
WINDOW_STRIDE = 20

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Label Map
try:
    with open(LABEL_MAP_PATH, 'r') as f:
        idx_to_label = json.load(f)
        ACTIONS = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    logging.info(f"Successfully loaded label map. Actions: {ACTIONS}")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load label map. Exiting. Error: {e}", exc_info=True)
    sys.exit(1)

# Load the TensorFlow Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    logging.info(f"TensorFlow model loaded and warmed up from {MODEL_PATH}")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load Keras model. Exiting. Error: {e}", exc_info=True)
    sys.exit(1)


class LstmPredictionService(prediction_services_pb2_grpc.LstmServiceServicer):
    """
    Implements the gRPC LstmService as defined in the .proto file.
    Handles predictions using a pre-trained TensorFlow LSTM model.
    """
    def Predict(self, request, context):
        """
        Handles the unary RPC for predicting multiple gestures from a list of GestureSequence messages.
        """
        try:
            # The client now sends one long gesture sequence.
            if not request.gestures:
                return prediction_services_pb2.LstmResponse(translated_text="")
            
            long_sequence_frames = request.gestures[0].frames
            num_frames = len(long_sequence_frames)
            logging.info(f"Received a continuous stream with {num_frames} frames.")

            if num_frames < SEQUENCE_LENGTH:
                logging.warning(f"Recording is too short ({num_frames} frames) to make a prediction. Needs at least {SEQUENCE_LENGTH}.")
                return prediction_services_pb2.LstmResponse(translated_text="Recording too short.")

            # Sliding Window Implementation
            windows_to_predict = []
            for i in range(0, num_frames - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
                window = long_sequence_frames[i : i + SEQUENCE_LENGTH]
                
                # Flatten the keypoints in the window into a single list
                current_window_data = []
                for frame in window:
                    current_window_data.extend(frame.keypoints)
                
                # Reshape and append for batch prediction
                np_window = np.array(current_window_data, dtype=np.float32).reshape(SEQUENCE_LENGTH, FEATURE_SIZE)
                windows_to_predict.append(np_window)
            
            if not windows_to_predict:
                 return prediction_services_pb2.LstmResponse(translated_text="")

            # Batch Prediction
            logging.info(f"Predicting on {len(windows_to_predict)} windows...")
            batch_input_np = np.array(windows_to_predict)
            predictions = model.predict(batch_input_np, verbose=0)
            predicted_indices = np.argmax(predictions, axis=1)
            predicted_actions_per_window = [ACTIONS[idx] for idx in predicted_indices]

            # Post-processing: Merge duplicate consecutive predictions
            if not predicted_actions_per_window:
                return prediction_services_pb2.LstmResponse(translated_text="")
            final_sentence = [predicted_actions_per_window[0]]
            for action in predicted_actions_per_window[1:]:
                if action != final_sentence[-1]:
                    final_sentence.append(action)
            
            response_text = " ".join(final_sentence)
            logging.info(f"Final translated sentence: '{response_text}'")
            return prediction_services_pb2.LstmResponse(translated_text=response_text)

        except Exception as e:
            logging.error(f"Error in Predict RPC: {e}", exc_info=True)
            context.set_details(f"Internal server error: {type(e).__name__}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return prediction_services_pb2.LstmResponse(translated_text="Error processing gesture prediction.")


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
    logging.info(f"LSTM gRPC Server started on port 50051. Using sliding window with stride {WINDOW_STRIDE}.")

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server shutting down.")
        server.stop(0)
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        server.stop(1)
        sys.exit(1)


if __name__ == "__main__":
    serve()