import os
import sys
import logging
import uuid
import grpc
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import your utility and protobufs
from utils.gesture_utils import process_video_parallel
import sign_data_nlp_pb2
import sign_data_nlp_pb2_grpc
import sign_data_lstm_pb2
import sign_data_lstm_pb2_grpc

app = Flask(__name__)
CORS(app)

# Configuration from environment variables
debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't') # Default to False for production
lstm_host = os.getenv("LSTM_HOST", "localhost:50051")
nlp_host = os.getenv("NLP_HOST", "localhost:50052")
app.config['DEBUG'] = debug_mode

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO if not debug_mode else logging.DEBUG, # More verbose in debug
    stream=sys.stdout
)

# Constants
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024 # 1GB
SEQUENCE_LENGTH = 80
HOLISTIC_MODEL_COMPLEXITY = 1

@app.route("/upload", methods=["POST"])
def upload():
    logging.info("Video upload request received.")

    # 1. Video Reception and Validation
    if 'video' not in request.files:
        logging.warning("No 'video' file part in the request.")
        return jsonify({"error": "No video file part"}), 400

    video = request.files["video"]
    if video.filename == '':
        logging.warning("No selected video file.")
        return jsonify({"error": "No selected file"}), 400

    # Secure filename to prevent directory traversal attacks
    filename = secure_filename(video.filename)
    if not filename.lower().endswith(('.webm', '.mp4', '.mov', '.avi')):
        logging.warning(f"Unsupported file format received: {filename}")
        return jsonify({"error": "Unsupported video format. Only .webm, .mp4, .mov, .avi are supported."}), 415 # Unsupported Media Type

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
        # 2. Video Processing (Keypoint Extraction)
        logging.info(f"Starting video processing for {video_path}...")
        sequences = process_video_parallel(
            sequence_length=SEQUENCE_LENGTH,
            model_complexity=HOLISTIC_MODEL_COMPLEXITY,
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
        # Ensure temporary video file is removed regardless of success or failure
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logging.info(f"Removed temporary video file: {video_path}")
            except OSError as cleanup_e:
                logging.warning(f"Error removing temporary video file {video_path}: {cleanup_e}")

    translated_text = "Processing failed"
    # 3. Call LSTM gRPC Service
    try:
        logging.info("Sending sequences to LSTM gRPC service...")
        # Create a single Gesture object with all keypoints flattened, as per LSTM service design for continuous stream
        # Each sequence is (SEQUENCE_LENGTH, feature_size) -> flatten to 1D array of feature_size * SEQUENCE_LENGTH
        # Then, if you send multiple sequences (each as a Gesture message), the LSTM app handles it as a batch.
        # If you send one large Gesture with all flattened points, the LSTM app uses the sliding window.
        
        # To align with the LSTM service's 'single long stream' logic, we concatenate all sequences into one `Gesture`
        # if the total length is greater than SEQUENCE_LENGTH, otherwise, send as individual gestures.
        
        # If the input sequences list from process_video_parallel is already segmented by SEQUENCE_LENGTH,
        # then each item in 'sequences' is already one full 'sequence'.
        # The LSTM `app.py` handles `len(request.data) == 1` for a continuous stream (sliding window)
        # and `len(request.data) > 1` for isolated gestures (batch inference).
        # Your `process_video_parallel` extracts non-overlapping `SEQUENCE_LENGTH` chunks.
        # So it's more accurate to send each chunk as an individual `Gesture` for batching on the LSTM side.
        
        # If process_video_parallel returns a list of (SEQUENCE_LENGTH, feature_size) arrays:
        gestures_for_lstm = []
        for seq_arr in sequences:
            # Flatten the numpy array for protobuf
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
    # 4. Call NLP gRPC Service
    try:
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
    logging.warning("Running Flask server. For production, use a production WSGI server like Gunicorn or uWSGI.")
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)