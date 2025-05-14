from utils.gesture_utils import process_video_parallel
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

app = Flask(__name__)
CORS(app)

debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
lstm_host = os.getenv("LSTM_HOST", "localhost:50051")
nlp_host = os.getenv("NLP_HOST", "localhost:50052")
app.config['DEBUG'] = debug_mode

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
SEQUENCE_LENGTH = 80
HOLISTIC_MODEL_COMPLEXITY = 1

@app.route("/upload", methods=["POST"])
def upload():
    logging.info("Video upload request received.")

    video = request.files.get("video")
    if not video or video.filename == '':
        logging.warning("No video file uploaded.")
        return jsonify({"error": "No file uploaded"}), 400

    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", f"{uuid.uuid4()}.webm")
    logging.info(f"Saving video to: {video_path}")
    try:
        video.save(video_path)
        logging.info("Video saved successfully.")
    except Exception as e:
        logging.error(f"Error saving video file: {e}")
        return jsonify({"error": "Failed to save video file"}), 500

    logging.info(f"Starting video processing for {video_path}...")
    try:
        sequences = process_video_parallel(
            sequence_length=SEQUENCE_LENGTH,
            model_complexity=HOLISTIC_MODEL_COMPLEXITY,
            video_path=video_path
        )
        logging.info(f"Video processing finished. Obtained {len(sequences)} sequences.")

        try:
            os.remove(video_path)
            logging.info(f"Removed temporary video file: {video_path}")
        except OSError as e:
            logging.warning(f"Error removing temporary video file {video_path}: {e}")

        if not sequences:
            logging.warning("Video processing produced no sequences.")
            return jsonify({"error": "Could not extract sequences from video"}), 500

    except Exception as e:
        logging.error(f"An error occurred during video processing: {e}", exc_info=True)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logging.info(f"Removed temporary video file after error: {video_path}")
        except OSError as cleanup_e:
             logging.warning(f"Error removing temporary video file during error cleanup {video_path}: {cleanup_e}")
        return jsonify({"error": f"Video processing failed: {e}"}), 500

    logging.info("Sending sequences to LSTM gRPC service...")
    translated_text = "Processing failed"
    try:
        with grpc.insecure_channel(lstm_host, options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub = sign_data_lstm_pb2_grpc.StreamDataServiceStub(channel)
            gestures = [sign_data_lstm_pb2.Gesture(points=[value for frame_kp_list in sequence for value in frame_kp_list]) for sequence in sequences]
            request_message = sign_data_lstm_pb2.RequestMessage(data=gestures)
            response = stub.biDirectionalStream(request_message)
            translated_text = response.reply
            logging.info(f"Received translated text from LSTM: '{translated_text}'")
    except grpc.RpcError as e:
        logging.error(f"LSTM gRPC call failed: {e}", exc_info=True)
        return jsonify({"error": f"LSTM service call failed: {e.code().name}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during LSTM gRPC call: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during LSTM call: {e}"}), 500

    logging.info("Sending translated text to NLP gRPC service...")
    nlp_response_text = translated_text
    try:
        with grpc.insecure_channel(nlp_host, options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub = sign_data_nlp_pb2_grpc.StreamDataServiceStub(channel)
            request_message = sign_data_nlp_pb2.RequestMessage(data=translated_text)
            response = stub.biDirectionalStream(request_message)
            nlp_response_text = response.reply
            logging.info(f"Received refined text from NLP: '{nlp_response_text}'")
    except grpc.RpcError as e:
        logging.error(f"NLP gRPC call failed: {e}", exc_info=True)
        return jsonify({"error": f"NLP service call failed: {e.code().name}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during NLP gRPC call: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during NLP call: {e}"}), 500

    logging.info("Request processing finished successfully.")
    return jsonify({"translatedText": nlp_response_text})

if __name__ == '__main__':
    logging.warning("Running development server. Use a production WSGI server for production.")
    app.run(host="0.0.0.0", debug=debug_mode, port=8080)