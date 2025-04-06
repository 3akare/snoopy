from flask import Flask, request, jsonify
from flask_cors import CORS
import grpc
import sign_data_pb2
import sign_data_pb2_grpc
import os

app = Flask(__name__)

# Get the environment variable for debug mode
debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
app.config['DEBUG'] = debug_mode

CORS(app)

@app.route('/nlp', methods=['POST'])
def nlp():
    data = request.get_json()  # Ensure JSON parsing
    query = data.get("query", "Default message")  # Extract 'query' safely
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = sign_data_pb2_grpc.StreamDataServiceStub(channel)
        # Send the query as a string in RequestMessage
        request_message = sign_data_pb2.RequestMessage(data=query)
        response = stub.biDirectionalStream(request_message)
        return f"[nlp] {response.reply}"

@app.route('/lstm', methods=['POST'])
def lstm():
    data = request.get_json()  # Ensure JSON parsing
    query = data.get("query", "Default message")  # Extract 'query' safely
    with grpc.insecure_channel("localhost:50052") as channel:
        stub = sign_data_pb2_grpc.StreamDataServiceStub(channel)
        # Send the query as a string in RequestMessage
        request_message = sign_data_pb2.RequestMessage(data=query)
        response = stub.biDirectionalStream(request_message)
        return f"[lstm] {response.reply}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=debug_mode, port=3000)
