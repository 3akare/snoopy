import os
import sys
import logging
from concurrent import futures
import numpy as np
import tensorflow as tf
import grpc

import sign_data_lstm_pb2
import sign_data_lstm_pb2_grpc

# Configuration constants
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1GB
ACTIONS = ["D", "A", "V", "I", "ME", "NAME"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')
SEQUENCE_LENGTH = 80 # Define it here for consistency

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

# Load model once at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("TensorFlow model loaded successfully.")
    # Warm-up the model with a dummy prediction to compile graph
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, model.input_shape[-1]), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    logging.info("Model warmed up.")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load TensorFlow model from {MODEL_PATH}. Exiting. Error: {e}")
    sys.exit(1) # Exit if model cannot be loaded

class StreamDataService(sign_data_lstm_pb2_grpc.StreamDataServiceServicer):
    def biDirectionalStream(self, request, context):
        try:
            if not request.data:
                logging.warning("Received request with no gesture data.")
                return sign_data_lstm_pb2.ResponseMessage(reply="No gesture data received")

            all_sequences_to_predict = []
            
            # Determine processing mode: single long stream or multiple isolated gestures
            if len(request.data) == 1:
                gesture = request.data[0]
                total_points = len(gesture.points)

                if total_points == 0:
                    logging.warning("Received empty points array in a single gesture request.")
                    return sign_data_lstm_pb2.ResponseMessage(reply="Empty gesture data received")

                # Infer feature_size from the first frame's expected structure based on the model's input
                # Assuming the model expects a fixed feature size (e.g., 1662 keypoints as previously seen)
                # This needs to be consistent with the `extract_keypoints` output in `gesture_utils.py`
                # If feature_size is not fixed, the model input_shape needs to be dynamic or a different strategy is needed.
                # For `extract_keypoints` (lh + rh) = 21*3 + 21*3 = 126
                # If pose + face + lh + rh, it's 33*4 + 468*3 + 21*3 + 21*3 = 132 + 1404 + 63 + 63 = 1662
                
                # We expect the input from `gesture_utils.py` to be `lh` + `rh` (21*3 + 21*3 = 126)
                feature_size = model.input_shape[-1] 
                
                if total_points % feature_size != 0:
                    logging.error(f"Total points ({total_points}) is not divisible by inferred feature size ({feature_size}). Invalid data shape.")
                    return sign_data_lstm_pb2.ResponseMessage(reply="Invalid data shape for prediction.")

                total_frames = total_points // feature_size
                data_reshaped = np.array(gesture.points, dtype=np.float32).reshape((total_frames, feature_size))

                if total_frames < SEQUENCE_LENGTH:
                    logging.warning(f"Single gesture data too short: {total_frames} frames, need at least {SEQUENCE_LENGTH}. Skipping prediction.")
                    return sign_data_lstm_pb2.ResponseMessage(reply="Data too short for prediction.")

                # Sliding window over frames with a fixed step
                step = SEQUENCE_LENGTH // 2 # 50% overlap for better coverage, adjust as needed
                if step == 0: step = 1 # Ensure at least 1 step
                
                for start in range(0, total_frames - SEQUENCE_LENGTH + 1, step):
                    window = data_reshaped[start : start + SEQUENCE_LENGTH]
                    all_sequences_to_predict.append(window)

            else: # Multiple isolated gestures (batch inference)
                # Validate consistency of input sequences if multiple are provided
                first_gesture = request.data[0]
                expected_points_per_sequence = SEQUENCE_LENGTH * model.input_shape[-1]
                
                if len(first_gesture.points) != expected_points_per_sequence:
                     logging.error(f"First gesture has {len(first_gesture.points)} points, expected {expected_points_per_sequence} (SEQUENCE_LENGTH * feature_size).")
                     return sign_data_lstm_pb2.ResponseMessage(reply="Inconsistent gesture data length.")
                
                feature_size = model.input_shape[-1] # From the model's expected input

                for i, gesture in enumerate(request.data):
                    if len(gesture.points) != expected_points_per_sequence:
                        logging.error(f"Inconsistent points in gesture {i}: expected {expected_points_per_sequence}, got {len(gesture.points)}.")
                        # It's better to process valid ones and log errors for invalid,
                        # but given the current flow, it's safer to reject the whole batch.
                        return sign_data_lstm_pb2.ResponseMessage(reply=f"Inconsistent points in gesture {i}. All gestures in a batch must have same length.")
                    
                    seq = np.array(gesture.points, dtype=np.float32).reshape((SEQUENCE_LENGTH, feature_size))
                    all_sequences_to_predict.append(seq)
            
            if not all_sequences_to_predict:
                logging.warning("No valid sequences constructed for prediction.")
                return sign_data_lstm_pb2.ResponseMessage(reply="No valid sequences to predict.")

            sequences_np = np.array(all_sequences_to_predict)
            
            # Perform prediction
            predictions = model.predict(sequences_np, verbose=0) # Suppress verbose output for cleaner logs
            
            # Map predictions to actions
            predicted_action_indices = np.argmax(predictions, axis=1)
            predicted_actions = [ACTIONS[idx] for idx in predicted_action_indices]

            final_response_text = ""
            if len(request.data) == 1: # If processing a single long video stream
                # Post-processing for continuous stream: merge and filter duplicates
                # This is a simple run-length encoding type filter
                filtered_actions = []
                if predicted_actions:
                    filtered_actions.append(predicted_actions[0])
                    for i in range(1, len(predicted_actions)):
                        if predicted_actions[i] != predicted_actions[i-1]:
                            filtered_actions.append(predicted_actions[i])
                final_response_text = " ".join(filtered_actions)
            else: # If processing multiple isolated gestures
                final_response_text = " ".join(predicted_actions)
            
            logging.info(f"Translated text: '{final_response_text}'")
            return sign_data_lstm_pb2.ResponseMessage(reply=final_response_text)

        except Exception as e:
            logging.error(f"Error in biDirectionalStream: {e}", exc_info=True)
            return sign_data_lstm_pb2.ResponseMessage(reply="Error processing gesture.")

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2), # Optimal workers for I/O bound tasks
        options=[
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    sign_data_lstm_pb2_grpc.add_StreamDataServiceServicer_to_server(StreamDataService(), server)
    server.add_insecure_port("[::]:50051")
    logging.info("LSTM gRPC Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()