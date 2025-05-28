import os
import sys
import logging
import uuid
import grpc
import sign_data_nlp_pb2
import sign_data_lstm_pb2
import sign_data_nlp_pb2_grpc
import sign_data_lstm_pb2_grpc
from flask_cors import CORS
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils.gesture_utils import process_video_parallel

app = Flask(__name__)
CORS(app)

debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
lstm_host = os.getenv("LSTM_HOST", "localhost:50051")
nlp_host = os.getenv("NLP_HOST", "localhost:50052")
app.config['DEBUG'] = debug_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
SEQUENCE_LENGTH = 80

@app.route("/upload", methods=["POST"])
def upload():
    logging.info("Video upload request received.")
    if 'video' not in request.files:
        logging.warning("No 'video' file part in the request.")
        return jsonify({"error": "No video file part"}), 400

    video = request.files["video"]
    if video.filename == '':
        logging.warning("No selected video file.")
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(video.filename)
    if not filename.lower().endswith(('.webm', '.mp4', '.mov', '.avi')):
        logging.warning(f"Unsupported file format received: {filename}")
        return jsonify({"error": "Unsupported video format. Only .webm, .mp4, .mov, .avi are supported."}), 415
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{filename}")
    
    try:
        video.save(video_path)
        logging.info(f"Video saved to: {video_path}")
    except Exception as e:
        logging.error(f"Error saving video file to {video_path}: {e}", exc_info=True)
        return jsonify({"error": "Failed to save video file"}), 500

    sequences = []
    try:
        logging.info(f"Starting video processing for {video_path}...")
        sequences = process_video_parallel(
            sequence_length=SEQUENCE_LENGTH,
            video_path=video_path
        )
        logging.info(f"Video processing finished. Extracted {len(sequences)} sequences.")

        if not sequences:
            logging.warning(f"Video processing for {video_path} produced no sequences.")
            return jsonify({"error": "Could not extract sequences from video. Video might be too short or lack detectable features."}), 500

    except Exception as e:
        logging.error(f"An error occurred during video processing of {video_path}: {e}", exc_info=True)
        return jsonify({"error": f"Video processing failed: {e}"}), 500
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logging.info(f"Removed temporary video file: {video_path}")
            except OSError as cleanup_e:
                logging.warning(f"Error removing temporary video file {video_path}: {cleanup_e}")
    translated_text = "Processing failed"
    try:
        # LSTM gRPC call
        logging.info("Sending sequences to LSTM gRPC service...")
        gestures_for_lstm = []
        for seq_arr in sequences:
            gestures_for_lstm.append(sign_data_lstm_pb2.Gesture(points=seq_arr.flatten().tolist()))

        request_message_lstm = sign_data_lstm_pb2.RequestMessage(data=gestures_for_lstm)
        
        with grpc.insecure_channel(lstm_host, options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub_lstm = sign_data_lstm_pb2_grpc.StreamDataServiceStub(channel)
            response_lstm = stub_lstm.biDirectionalStream(request_message_lstm)
            translated_text = response_lstm.reply
            logging.info(f"Received translated text from LSTM: '{translated_text}'")
    except grpc.RpcError as e:
        logging.error(f"LSTM gRPC call failed: {e.details()} (Code: {e.code().name})", exc_info=True)
        return jsonify({"error": f"LSTM service call failed: {e.code().name} - {e.details()}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during LSTM gRPC call: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during LSTM call: {e}"}), 500

    nlp_response_text = translated_text
    try:
        # NLP gRPC call
        logging.info("Sending translated text to NLP gRPC service...")
        request_message_nlp = sign_data_nlp_pb2.RequestMessage(data=translated_text)
        with grpc.insecure_channel(nlp_host, options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub_nlp = sign_data_nlp_pb2_grpc.StreamDataServiceStub(channel)
            response_nlp = stub_nlp.biDirectionalStream(request_message_nlp)
            nlp_response_text = response_nlp.reply
            logging.info(f"Received refined text from NLP: '{nlp_response_text}'")
    except grpc.RpcError as e:
        logging.error(f"NLP gRPC call failed: {e.details()} (Code: {e.code().name})", exc_info=True)
        return jsonify({"error": f"NLP service call failed: {e.code().name} - {e.details()}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during NLP gRPC call: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during NLP call: {e}"}), 500

    logging.info("Request processing finished successfully.")
    return jsonify({"translatedText": nlp_response_text})

if __name__ == '__main__':
    logging.warning("Running Flask server...")
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)