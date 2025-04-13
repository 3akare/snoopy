from utils.gesture_utils import process_video
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import grpc
import logging
import uuid
import sign_data_nlp_pb2
import sign_data_nlp_pb2_grpc
import sign_data_lstm_pb2
import sign_data_lstm_pb2_grpc
import mediapipe as mp

# Flask app setup
app = Flask(__name__)
CORS(app)

# Environment variables and configuration
debug_mode = os.getenv('FLASK_DEBUG', 'True')
lstm_host = os.getenv("LSTM_HOST", "localhost:50051")
nlp_host = os.getenv("NLP_HOST", "localhost:50052")
app.config['DEBUG'] = debug_mode

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

# Constants
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
SEQUENCE_LENGTH = 30
HOLISTIC = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.route("/upload", methods=["POST"])
def upload():
    logging.info("Video received\nProcessing...")
    
    # Get the uploaded video
    video = request.files.get("video")
    if not video or video.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded video
    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", f"{uuid.uuid4()}.webm")
    logging.info(f"File path: {video_path}")
    video.save(video_path)

    # Process the video and get sequences
    sequences = process_video(SEQUENCE_LENGTH, HOLISTIC, video_path)

    # Send sequences to LSTM for gesture translation
    with grpc.insecure_channel(lstm_host, options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)]) as channel:
        stub = sign_data_lstm_pb2_grpc.StreamDataServiceStub(channel)
        gestures = [sign_data_lstm_pb2.Gesture(points=[value for frame in sequence for value in frame]) for sequence in sequences]
        request_message = sign_data_lstm_pb2.RequestMessage(data=gestures)
        response = stub.biDirectionalStream(request_message)
        translated_text = response.reply
        logging.info(f"Translated text: {translated_text}")

        # Send translated text to NLP service for refinement
        with grpc.insecure_channel(nlp_host, options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub = sign_data_nlp_pb2_grpc.StreamDataServiceStub(channel)
            request_message = sign_data_nlp_pb2.RequestMessage(data=translated_text)
            response = stub.biDirectionalStream(request_message)
            logging.info(f"[nlp] {response.reply}\nDone.")

            return jsonify({"translatedText": response.reply})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=debug_mode, port=8080)
