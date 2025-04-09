from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import grpc
import logging
import uuid
import sign_data_nlp_pb2
import sign_data_nlp_pb2_grpc
import sign_data_lstm_pb2
import sign_data_lstm_pb2_grpc

import os

app = Flask(__name__)

# Get the environment variable for debug mode
debug_mode = os.getenv('FLASK_DEBUG', 'True')
lstm_host = os.getenv("LSTM_HOST", "localhost:50051")
nlp_host = os.getenv("NLP_HOST", "localhost:50052")

app.config['DEBUG'] = debug_mode

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

CORS(app)

@app.route("/upload", methods=["POST"])
def upload():
    video = request.files.get("video")
    if not video or video.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", str(uuid.uuid4()) + ".webm")
    logging.info(f"File path: {filepath}")
    video.save(filepath)
    translated_text = "Finally, the video got to the server. We are making good progress aren't we?"
    logging.info(f"Translated text: {translated_text}")
    return jsonify({"translatedText": translated_text})

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
    app.run(host="0.0.0.0", debug=debug_mode, port=8080)
