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
model = tf.keras.models.load_model(MODEL_PATH)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

class StreamDataService(sign_data_lstm_pb2_grpc.StreamDataServiceServicer):
    def biDirectionalStream(self, request, context):
        # logging.info(f"Received request data: {request.data}")
        print(f"Received request data: {request.data}")
        try:
            sequences = []
            for gesture in request.data:
                # Each gesture.points is a flattened sequence.
                total_length = len(gesture.points)
                sequence_length = 30  # Must match your capture
                feature_size = total_length // sequence_length
                sequence = np.array(gesture.points).reshape((sequence_length, feature_size))
                sequences.append(sequence)
            
            if not sequences:
                return sign_data_lstm_pb2.ResponseMessage(reply="No gesture data received")
            
            sequences = np.array(sequences)
            predictions = model.predict(sequences)
            predicted_actions = [label_map[int(np.argmax(pred))] for pred in predictions]
            response_text = " ".join(predicted_actions)
            # logging.info(f"Processed response: {response_text}")
            print(f"Processed response: {response_text}")
            return sign_data_lstm_pb2.ResponseMessage(reply=response_text)
        except Exception as e:
            print(f"Error in biDirectionalStream: {e}")
        return sign_data_lstm_pb2.ResponseMessage(reply=response_text)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), 
    options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)])

    sign_data_lstm_pb2_grpc.add_StreamDataServiceServicer_to_server(StreamDataService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    # logging.info("Server started on port 50051")
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
