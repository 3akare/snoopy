from flask import Flask, request, jsonify
from flask_cors import CORS
import grpc
import sign_data_nlp_pb2
import sign_data_nlp_pb2_grpc
import sign_data_lstm_pb2
import sign_data_lstm_pb2_grpc

import os

app = Flask(__name__)

# Get the environment variable for debug mode
debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
lstm_host = os.getenv("LSTM_HOST", "localhost:50051")
nlp_host = os.getenv("NLP_HOST", "localhost:50052")

app.config['DEBUG'] = debug_mode

CORS(app)

@app.route('/lstm', methods=['POST'])
def lstm():
    data = request.get_json()  # Ensure JSON parsing
    query = data.get("query", "Default message")  # Extract 'query' safely
    with grpc.insecure_channel(lstm_host) as channel:
        stub = sign_data_lstm_pb2_grpc.StreamDataServiceStub(channel)
        # Send the query as a string in RequestMessage
        request_message = sign_data_lstm_pb2.RequestMessage(data=[])
        response = stub.biDirectionalStream(request_message)
        return f"[lstm] {response.reply}"


@app.route('/nlp', methods=['POST'])
def nlp():
    data = request.get_json()  # Ensure JSON parsing
    query = data.get("query", "Default message")  # Extract 'query' safely
    with grpc.insecure_channel(nlp_host) as channel:
        stub = sign_data_nlp_pb2_grpc.StreamDataServiceStub(channel)
        # Send the query as a string in RequestMessage
        request_message = sign_data_nlp_pb2.RequestMessage(data=query)
        response = stub.biDirectionalStream(request_message)
        return f"[nlp] {response.reply}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=debug_mode, port=3000)
