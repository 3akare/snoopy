import cv2
import numpy as np
import mediapipe as mp
import concurrent.futures
import os
import logging
import sys

# Setup logging specific to this module
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

# Initialize MediaPipe Holistic once per process for efficiency
_worker_holistic_model = None

def _init_worker(model_complexity, min_detection_confidence, min_tracking_confidence):
    """Initializes MediaPipe Holistic model in each worker process."""
    global _worker_holistic_model
    try:
        _worker_holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False, # Process video frames (not static images)
            model_complexity=model_complexity
        )
        logging.info(f"Worker process {os.getpid()} initialized MediaPipe Holistic model (Complexity: {model_complexity}).")
    except Exception as e:
        logging.critical(f"Worker process {os.getpid()} failed to initialize MediaPipe Holistic model: {e}", exc_info=True)
        raise

def mediapipe_detection(image, model):
    """Processes image using MediaPipe Holistic model."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Image is read-only for MediaPipe
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results # Only return results, image conversion back to BGR is not needed by caller

def extract_keypoints(results):
    """Extracts keypoints from detected landmarks focusing on hands."""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    
    return np.concatenate([lh, rh]).astype(np.float32)

def _process_frame_task(frame_bytes_data):
    """
    Task for worker processes to decode a frame and extract keypoints.
    Receives frame as bytes to avoid shared memory issues and optimize serialization.
    """
    global _worker_holistic_model
    if _worker_holistic_model is None:
        logging.error(f"Worker {os.getpid()}: Holistic model not initialized. This indicates a setup issue.")
        return None

    try:
        # Correct unpacking: frame_bytes_data is expected to be a tuple (frame_array_bytes, frame_idx)
        frame_array_bytes, frame_idx = frame_bytes_data

        # Decode the image bytes into a NumPy array.
        frame_array = cv2.imdecode(np.frombuffer(frame_array_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame_array is None:
            logging.error(f"Worker process {os.getpid()} failed to decode frame {frame_idx}. Skipping.")
            return None

        # Define your target resolution.
        TARGET_WIDTH = 640
        TARGET_HEIGHT = 480

        # Check if resizing is needed and perform it.
        current_height, current_width, _ = frame_array.shape
        if current_width != TARGET_WIDTH or current_height != TARGET_HEIGHT:
            frame_array = cv2.resize(frame_array, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        # Mediapipe processing.
        results = mediapipe_detection(frame_array, _worker_holistic_model)
        keypoints = extract_keypoints(results)

        return keypoints

    except Exception as e:
        logging.error(f"Worker process {os.getpid()} failed to process frame: {e}", exc_info=True)
        return None

def process_video_parallel(sequence_length, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, video_path=None, frame_arrays=None):
    """
    Processes a video file or list of frames in parallel to extract keypoint sequences.
    
    Returns:
        list of numpy arrays, where each array is a sequence of shape (sequence_length, feature_size)
    """
    if video_path is None and frame_arrays is None:
        raise ValueError("Either video_path or frame_arrays must be provided.")

    frames_to_process_for_queue = [] # This will store tuples of (frame_bytes, frame_idx)
    
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
            
            # Send a tuple of (frame_bytes, frame_index)
            frames_to_process_for_queue.append((encoded_frame.tobytes(), frame_idx))
            frame_idx += 1
        cap.release()
        logging.info(f"Finished reading and encoding {len(frames_to_process_for_queue)} frames from video.")
    elif frame_arrays is not None:
        logging.info(f"Using {len(frame_arrays)} pre-loaded frames.")
        # Assuming frame_arrays are already numpy arrays, encode them
        for i, frame in enumerate(frame_arrays):
            ret_encode, encoded_frame = cv2.imencode('.png', frame)
            if not ret_encode:
                logging.warning(f"Failed to encode pre-loaded frame {i}. Skipping.")
                continue
            # Send a tuple of (frame_bytes, frame_index)
            frames_to_process_for_queue.append((encoded_frame.tobytes(), i))

    if not frames_to_process_for_queue:
        logging.warning("No frames available for processing.")
        return []

    logging.info(f"Starting parallel processing of {len(frames_to_process_for_queue)} frames...")
    all_keypoints = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=_init_worker,
        initargs=(model_complexity, min_detection_confidence, min_tracking_confidence)
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