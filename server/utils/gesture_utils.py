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
        # Re-raise to ensure the worker process fails gracefully
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
    # Ensure consistency with your model's expected input (e.g., 126 features for only hands)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32) # Ensure 0s are float32
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32) # Ensure 0s are float32
    
    # Concatenate in the order expected by your trained model
    return np.concatenate([lh, rh]).astype(np.float32)

def _process_frame_task(frame_array_bytes):
    """
    Task for worker processes to decode a frame and extract keypoints.
    Receives frame as bytes to avoid shared memory issues and optimize serialization.
    """
    global _worker_holistic_model
    if _worker_holistic_model is None:
        # This fallback might happen if initialization failed or for some other reason.
        # Ideally, _init_worker should prevent workers from starting without a model.
        logging.error(f"Worker {os.getpid()}: Holistic model not initialized. This indicates a setup issue.")
        return None

    try:
        # Decode the numpy array from bytes
        frame_array = np.frombuffer(frame_array_bytes, dtype=np.uint8).reshape(
            cv2.imdecode(np.frombuffer(frame_array_bytes, dtype=np.uint8), cv2.IMREAD_COLOR).shape
        )
        
        # Mediapipe processing
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

    frames_to_process = []
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {video_path}")
            return []
        
        logging.info(f"Reading frames from {video_path}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to bytes for efficient IPC (ProcessPoolExecutor uses pickle, which can be slow for large numpy arrays)
            # This creates a significant overhead during serialization/deserialization.
            # A more advanced solution would be shared memory or multiprocessing.Queue with raw bytes.
            # However, for simplicity and staying within current structure, we'll try this.
            
            # Using cv2.imencode for more robust serialization of image data
            ret_encode, encoded_frame = cv2.imencode('.png', frame) # Use PNG for lossless compression
            if not ret_encode:
                logging.warning(f"Failed to encode frame from {video_path}.")
                continue
            frames_to_process.append(encoded_frame.tobytes())
        cap.release()
        logging.info(f"Finished reading {len(frames_to_process)} frames from video.")
    elif frame_arrays is not None:
        logging.info(f"Using {len(frame_arrays)} pre-loaded frames.")
        # Assuming frame_arrays are already numpy arrays, encode them
        for frame in frame_arrays:
            ret_encode, encoded_frame = cv2.imencode('.png', frame)
            if not ret_encode:
                logging.warning("Failed to encode pre-loaded frame.")
                continue
            frames_to_process.append(encoded_frame.tobytes())

    if not frames_to_process:
        logging.warning("No frames available for processing.")
        return []

    logging.info(f"Starting parallel processing of {len(frames_to_process)} frames...")
    all_keypoints = []

    # Use ProcessPoolExecutor for CPU-bound MediaPipe processing
    # Max workers set to CPU count for optimal CPU utilization
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=_init_worker,
        initargs=(model_complexity, min_detection_confidence, min_tracking_confidence)
    ) as executor:
        try:
            # `chunksize` can optimize performance by sending frames in batches
            # to workers, reducing IPC overhead. Experiment with this value.
            # A common starting point is len(frames) // num_workers * 2 or similar.
            chunksize = max(1, len(frames_to_process) // (os.cpu_count() * 2)) 
            keypoints_iterator = executor.map(_process_frame_task, frames_to_process, chunksize=chunksize)
            all_keypoints = list(keypoints_iterator)
            logging.info("Finished parallel processing of frames.")
        except Exception as e:
            logging.error(f"An error occurred during parallel frame processing: {e}", exc_info=True)
            return []

    all_keypoints = [kp for kp in all_keypoints if kp is not None]
    if not all_keypoints:
        logging.error("No valid keypoints extracted from any frame. Check MediaPipe output or input video quality.")
        return []

    # Segment keypoints into sequences of fixed length
    logging.info(f"Segmenting {len(all_keypoints)} keypoint sets into sequences of length {sequence_length}...")
    sequences = []
    # Use a direct loop with a sliding window approach for segmentation,
    # ensuring each extracted sequence is exactly `sequence_length` long.
    # No buffering needed, just direct segmentation.
    
    # If the intent is to produce non-overlapping sequences for training data generation,
    # then iterate with step = sequence_length.
    # If it's for inference with a sliding window (like in train.py `process_video_and_save_sequences`),
    # that logic should be in the LSTM `app.py`.
    # Based on `train.py`, you extract non-overlapping sequences.
    
    for i in range(0, len(all_keypoints) - sequence_length + 1, sequence_length):
        sequence_segment = all_keypoints[i : i + sequence_length]
        sequences.append(np.array(sequence_segment)) # Store as numpy array

    logging.info(f"Finished segmentation. Created {len(sequences)} sequences.")
    return sequences