import os
import sys
import grpc
import logging
import numpy as np
import tensorflow as tf
import sign_data_lstm_pb2
import sign_data_lstm_pb2_grpc
from concurrent import futures

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
ACTIONS = [ "C", "D", "L", "R", "U", "V", "W", "X", "Y", "Z", "NAME"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.keras')
SEQUENCE_LENGTH = 80 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("TensorFlow model loaded successfully.")
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, model.input_shape[-1]), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    logging.info("Model warmed up.")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load TensorFlow model from {MODEL_PATH}. Exiting. Error: {e}")
    sys.exit(1)

class StreamDataService(sign_data_lstm_pb2_grpc.StreamDataServiceServicer):
    def biDirectionalStream(self, request, context):
        try:
            if not request.data:
                logging.warning("Received request with no gesture data.")
                return sign_data_lstm_pb2.ResponseMessage(reply="No gesture data received")
            all_sequences_to_predict = []
            if len(request.data) == 1:
                gesture = request.data[0]
                total_points = len(gesture.points)

                if total_points == 0:
                    logging.warning("Received empty points array in a single gesture request.")
                    return sign_data_lstm_pb2.ResponseMessage(reply="Empty gesture data received")
                feature_size = model.input_shape[-1] 
                if total_points % feature_size != 0:
                    logging.error(f"Total points ({total_points}) is not divisible by inferred feature size ({feature_size}). Invalid data shape.")
                    return sign_data_lstm_pb2.ResponseMessage(reply="Invalid data shape for prediction.")

                total_frames = total_points // feature_size
                data_reshaped = np.array(gesture.points, dtype=np.float32).reshape((total_frames, feature_size))

                if total_frames < SEQUENCE_LENGTH:
                    logging.warning(f"Single gesture data too short: {total_frames} frames, need at least {SEQUENCE_LENGTH}. Skipping prediction.")
                    return sign_data_lstm_pb2.ResponseMessage(reply="Data too short for prediction.")
                step = SEQUENCE_LENGTH // 2
                if step == 0: step = 1

                for start in range(0, total_frames - SEQUENCE_LENGTH + 1, step):
                    window = data_reshaped[start : start + SEQUENCE_LENGTH]
                    all_sequences_to_predict.append(window)

            else:
                first_gesture = request.data[0]
                expected_points_per_sequence = SEQUENCE_LENGTH * model.input_shape[-1]
                
                if len(first_gesture.points) != expected_points_per_sequence:
                     logging.error(f"First gesture has {len(first_gesture.points)} points, expected {expected_points_per_sequence} (SEQUENCE_LENGTH * feature_size).")
                     return sign_data_lstm_pb2.ResponseMessage(reply="Inconsistent gesture data length.")
                
                feature_size = model.input_shape[-1]

                for i, gesture in enumerate(request.data):
                    if len(gesture.points) != expected_points_per_sequence:
                        logging.error(f"Inconsistent points in gesture {i}: expected {expected_points_per_sequence}, got {len(gesture.points)}.")
                        return sign_data_lstm_pb2.ResponseMessage(reply=f"Inconsistent points in gesture {i}. All gestures in a batch must have same length.")
                    
                    seq = np.array(gesture.points, dtype=np.float32).reshape((SEQUENCE_LENGTH, feature_size))
                    all_sequences_to_predict.append(seq)
            
            if not all_sequences_to_predict:
                logging.warning("No valid sequences constructed for prediction.")
                return sign_data_lstm_pb2.ResponseMessage(reply="No valid sequences to predict.")
            sequences_np = np.array(all_sequences_to_predict)
            
            predictions = model.predict(sequences_np, verbose=0)
            predicted_action_indices = np.argmax(predictions, axis=1)
            predicted_actions = [ACTIONS[idx] for idx in predicted_action_indices]

            final_response_text = ""
            if len(request.data) == 1:
                filtered_actions = []
                if predicted_actions:
                    filtered_actions.append(predicted_actions[0])
                    for i in range(1, len(predicted_actions)):
                        if predicted_actions[i] != predicted_actions[i-1]:
                            filtered_actions.append(predicted_actions[i])
                final_response_text = " ".join(filtered_actions)
            else:
                final_response_text = " ".join(predicted_actions)
            logging.info(f"Translated text: '{final_response_text}'")
            return sign_data_lstm_pb2.ResponseMessage(reply=final_response_text)
        except Exception as e:
            logging.error(f"Error in biDirectionalStream: {e}", exc_info=True)
            return sign_data_lstm_pb2.ResponseMessage(reply="Error processing gesture.")

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2),
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