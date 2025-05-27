import os
import sys
import cv2
import logging
import numpy as np
import mediapipe as mp
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

_worker_hands_model = None

def _init_worker(min_detection_confidence, min_tracking_confidence):
    global _worker_hands_model
    try:
        _worker_hands_model = mp.solutions.hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False
        )
        logging.info(f"Worker process {os.getpid()} initialized MediaPipe Hands model.")
    except Exception as e:
        logging.critical(f"Worker process {os.getpid()} failed to initialize MediaPipe Hands model: {e}", exc_info=True)
        raise

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    return np.concatenate([lh, rh]).astype(np.float32)

def _process_frame_task(frame_bytes_data):
    global _worker_hands_model
    if _worker_hands_model is None:
        logging.error(f"Worker {os.getpid()}: Hands model not initialized. This indicates a setup issue.")
        return None
    try:
        frame_array_bytes, frame_idx = frame_bytes_data
        frame_array = cv2.imdecode(np.frombuffer(frame_array_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame_array is None:
            logging.error(f"Worker process {os.getpid()} failed to decode frame {frame_idx}. Skipping.")
            return None
        TARGET_WIDTH = 640
        TARGET_HEIGHT = 480
        current_height, current_width, _ = frame_array.shape
        if current_width != TARGET_WIDTH or current_height != TARGET_HEIGHT:
            frame_array = cv2.resize(frame_array, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        results = mediapipe_detection(frame_array, _worker_hands_model)
        keypoints = extract_keypoints(results)
        return keypoints

    except Exception as e:
        logging.error(f"Worker process {os.getpid()} failed to process frame: {e}", exc_info=True)
        return None

def process_video_parallel(sequence_length, min_detection_confidence=0.7, min_tracking_confidence=0.5, video_path=None, frame_arrays=None):
    if video_path is None and frame_arrays is None:
        raise ValueError("Either video_path or frame_arrays must be provided.")
    frames_to_process_for_queue = []
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {video_path}")
            return []
        logging.info(f"Reading frames from {video_path}...")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            ret_encode, encoded_frame = cv2.imencode('.png', frame)
            if not ret_encode:
                logging.warning(f"Failed to encode frame {frame_idx} from {video_path}. Skipping.")
                continue
            frames_to_process_for_queue.append((encoded_frame.tobytes(), frame_idx))
            frame_idx += 1
        cap.release()
        logging.info(f"Finished reading and encoding {len(frames_to_process_for_queue)} frames from video.")
    elif frame_arrays is not None:
        logging.info(f"Using {len(frame_arrays)} pre-loaded frames.")
        for i, frame in enumerate(frame_arrays):
            ret_encode, encoded_frame = cv2.imencode('.png', frame)
            if not ret_encode:
                logging.warning(f"Failed to encode pre-loaded frame {i}. Skipping.")
                continue
            frames_to_process_for_queue.append((encoded_frame.tobytes(), i))
    if not frames_to_process_for_queue:
        logging.warning("No frames available for processing.")
        return []
    logging.info(f"Starting parallel processing of {len(frames_to_process_for_queue)} frames...")
    all_keypoints = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=_init_worker,
        initargs=(min_detection_confidence, min_tracking_confidence)
    ) as executor:
        try:
            chunksize = max(1, len(frames_to_process_for_queue) // (os.cpu_count() * 2)) 
            keypoints_iterator = executor.map(_process_frame_task, frames_to_process_for_queue, chunksize=chunksize)
            all_keypoints = list(keypoints_iterator)
            logging.info("Finished parallel processing of frames.")
        except Exception as e:
            logging.error(f"An error occurred during parallel frame processing: {e}", exc_info=True)
            return []

    all_keypoints = [kp for kp in all_keypoints if kp is not None]
    if not all_keypoints:
        logging.error("No valid keypoints extracted from any frame. Check MediaPipe output or input video quality.")
        return []
    logging.info(f"Segmenting {len(all_keypoints)} keypoint sets into sequences of length {sequence_length}...")
    sequences = []
    for i in range(0, len(all_keypoints) - sequence_length + 1, sequence_length):
        sequence_segment = all_keypoints[i : i + sequence_length]
        sequences.append(np.array(sequence_segment))
    logging.info(f"Finished segmentation. Created {len(sequences)} sequences.")
    return sequences