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

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Label Map
try:
    with open(LABEL_MAP_PATH, 'r') as f:
        idx_to_label = json.load(f)
        # The model outputs indices, so we need a list where the index corresponds to the label.
        # If idx_to_label is {"0": "A", "1": "B"}, ACTIONS becomes ["A", "B"]
        ACTIONS = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    logging.info(f"Successfully loaded label map. Actions: {ACTIONS}")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load label map from {LABEL_MAP_PATH}. Exiting. Error: {e}", exc_info=True)
    sys.exit(1)

# Load and Warm Up the TensorFlow Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("TensorFlow model loaded successfully.")
    # Assuming the model's input shape is (None, SEQUENCE_LENGTH, FEATURE_SIZE)
    # The first dimension (None) is the batch size.
    # We extract the sequence length and feature size from the model's expected input.
    _, SEQUENCE_LENGTH, FEATURE_SIZE = model.input_shape
    logging.info(f"Model properties loaded: Sequence Length={SEQUENCE_LENGTH}, Feature Size={FEATURE_SIZE}")
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    logging.info("Model warmed up.")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load TensorFlow model from {MODEL_PATH}. Exiting. Error: {e}", exc_info=True)
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
            if not request.gestures:
                logging.warning("Received LstmRequest with no gesture sequences.")
                return prediction_services_pb2.LstmResponse(translated_text="No gesture sequences received")

            all_sequences_to_predict = []
            
            # Iterate through each distinct GestureSequence provided in the request
            for i, gesture_sequence in enumerate(request.gestures):
                
                # Check if the number of frames in the current gesture's sequence matches the model's SEQUENCE_LENGTH
                if len(gesture_sequence.frames) != SEQUENCE_LENGTH:
                    error_msg = (
                        f"Gesture {i}: Inconsistent number of frames. Expected {SEQUENCE_LENGTH}, "
                        f"got {len(gesture_sequence.frames)}."
                    )
                    logging.error(error_msg)
                    # If any single gesture is malformed, we might return an error for the whole request
                    context.set_details(f"Invalid input: {error_msg}")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return prediction_services_pb2.LstmResponse(translated_text=error_msg)

                # Prepare a list to hold all keypoints for the current single gesture's sequence
                current_gesture_flattened_data = []

                # Iterate through each KeypointFrame within the current GestureSequence
                for j, frame in enumerate(gesture_sequence.frames):
                    # Ensure each frame has the expected number of keypoints (FEATURE_SIZE)
                    if len(frame.keypoints) != FEATURE_SIZE:
                        error_msg = (
                            f"Gesture {i}, Frame {j}: Inconsistent keypoints. Expected {FEATURE_SIZE}, "
                            f"got {len(frame.keypoints)}."
                        )
                        logging.error(error_msg)
                        context.set_details(f"Invalid input: {error_msg}")
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        return prediction_services_pb2.LstmResponse(translated_text=error_msg)
                    
                    current_gesture_flattened_data.extend(frame.keypoints)
                
                # Reshape the flattened data for the current gesture into (SEQUENCE_LENGTH, FEATURE_SIZE)
                # And add it to the list of all sequences for batch prediction
                all_sequences_to_predict.append(
                    np.array(current_gesture_flattened_data, dtype=np.float32).reshape((SEQUENCE_LENGTH, FEATURE_SIZE))
                )
            
            if not all_sequences_to_predict:
                logging.warning("No valid gesture sequences to predict after parsing.")
                return prediction_services_pb2.LstmResponse(translated_text="No valid gesture sequences to predict.")

            # Convert list of all gesture sequences to a NumPy array for batch prediction
            # The shape will be (num_gestures, SEQUENCE_LENGTH, FEATURE_SIZE)
            batch_input_np = np.array(all_sequences_to_predict)
            
            # Perform batch prediction
            predictions = model.predict(batch_input_np, verbose=0)
            
            # Get the index of the highest probability for each prediction in the batch
            predicted_indices = np.argmax(predictions, axis=1)
            
            # Map the predicted indices to their corresponding action labels
            predicted_actions = [ACTIONS[idx] for idx in predicted_indices]

            # Join all predicted actions into a single string
            response_text = " ".join(predicted_actions)
            logging.info(f"Translated text: '{response_text}' for {len(predicted_actions)} gestures.")
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
    # Register the LstmPredictionService with the gRPC server
    prediction_services_pb2_grpc.add_LstmServiceServicer_to_server(LstmPredictionService(), server)
    server.add_insecure_port("[::]:50051")
    logging.info("LSTM gRPC Server started on port 50051")

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("LSTM gRPC Server shutting down gracefully.")
        server.stop(0)
    except Exception as e:
        logging.critical(f"LSTM gRPC Server crashed: {e}", exc_info=True)


if __name__ == "__main__":
    serve()