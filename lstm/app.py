from concurrent import futures
import os
import sys
import logging
import grpc
import numpy as np
import tensorflow as tf
import sign_data_lstm_pb2
import sign_data_lstm_pb2_grpc

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
ACTIONS = ["hello", "thanks", "i love you"]
label_map = {i: action for i, action in enumerate(ACTIONS)}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    logging.error(f"Error loading model: {e}")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

class StreamDataService(sign_data_lstm_pb2_grpc.StreamDataServiceServicer):
    def biDirectionalStream(self, request, context):
        print(f"Received request with {len(request.data)} gestures.")
        try:
            if not request.data:
                print("No gesture data received")
                return sign_data_lstm_pb2.ResponseMessage(reply="No gesture data received")

            num_gestures = len(request.data)
            sequence_length = 30
            all_points_flat = []
            feature_size = None
            total_points_per_gesture = None

            if not request.data[0].points:
                logging.error("Invalid gesture data: First gesture has no points.")
                return sign_data_lstm_pb2.ResponseMessage(reply="Invalid gesture data: First gesture is empty")

            total_points_per_gesture = len(request.data[0].points)
            if total_points_per_gesture % sequence_length != 0:
                 logging.error(f"Invalid gesture data: First gesture total points ({total_points_per_gesture}) not divisible by sequence length ({sequence_length}).")
                 return sign_data_lstm_pb2.ResponseMessage(reply=f"Invalid gesture data: Point count ({total_points_per_gesture}) not divisible by sequence length ({sequence_length})")

            feature_size = total_points_per_gesture // sequence_length

            for i, gesture in enumerate(request.data):
                if len(gesture.points) != total_points_per_gesture:
                    logging.error(f"Invalid gesture data: Gesture {i} has inconsistent number of points ({len(gesture.points)}), expected {total_points_per_gesture}.")
                    return sign_data_lstm_pb2.ResponseMessage(reply=f"Invalid gesture data: Inconsistent points in gesture {i}")
                all_points_flat.extend(gesture.points)

            try:
                 sequences = np.array(all_points_flat).reshape((num_gestures, sequence_length, feature_size))

            except ValueError as e:
                 logging.error(f"Error reshaping data: {e}. Expected total elements {num_gestures * sequence_length * feature_size}, got {len(all_points_flat)}.")
                 return sign_data_lstm_pb2.ResponseMessage(reply=f"Error processing data shape: {e}")

            predictions = model.predict(sequences)
            predicted_actions = [label_map[int(np.argmax(pred))] for pred in predictions]
            response_text = " ".join(predicted_actions)
            print(f"Processed response: {response_text}")
            return sign_data_lstm_pb2.ResponseMessage(reply=response_text)

        except Exception as e:
            logging.error(f"Error in biDirectionalStream: {e}", exc_info=True)
            return sign_data_lstm_pb2.ResponseMessage(reply="Error processing gesture")

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)
        ]
    )
    sign_data_lstm_pb2_grpc.add_StreamDataServiceServicer_to_server(StreamDataService(), server)
    server.add_insecure_port("[::]:50051")
    print("Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()