import cv2
import numpy as np
import mediapipe as mp
import concurrent.futures
import os
import logging
import sys

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

_worker_holistic_model = None
def _init_worker(model_complexity, min_detection_confidence, min_tracking_confidence):
    global _worker_holistic_model
    _worker_holistic_model = mp.solutions.holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        static_image_mode=False,
        model_complexity=model_complexity
    )
    logging.info(f"Worker process {os.getpid()} initialized MediaPipe Holistic model.")

def _process_frame_task(frame_array):
    global _worker_holistic_model
    if _worker_holistic_model is None:
        logging.warning(f"Worker process {os.getpid()}: Model not initialized, attempting fallback init.")
        _init_worker(1, 0.5, 0.5)
    try:
        _, results = mediapipe_detection(frame_array, _worker_holistic_model)
        keypoints = extract_keypoints(results)
        return keypoints
    except Exception as e:
        logging.error(f"Worker process {os.getpid()} failed to process frame: {e}")
        return None

def process_video_parallel(sequence_length, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, video_path=None, frame_arrays=None):
    if video_path is None and frame_arrays is None:
        raise ValueError("Either video_path or frame_arrays must be provided.")
    frames = []
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
            frames.append(frame)
        cap.release()
        logging.info(f"Finished reading {len(frames)} frames.")
    elif frame_arrays is not None:
        frames = frame_arrays
        logging.info(f"Using {len(frames)} pre-loaded frames.")
    if not frames:
        logging.warning("No frames to process.")
        return []
    logging.info(f"Starting parallel processing of {len(frames)} frames...")
    all_keypoints = []
    with concurrent.futures.ProcessPoolExecutor(
        initializer=_init_worker,
        initargs=(model_complexity, min_detection_confidence, min_tracking_confidence)
    ) as executor:
        try:
            keypoints_iterator = executor.map(_process_frame_task, frames)
            all_keypoints = list(keypoints_iterator)
            logging.info("Finished parallel processing.")
        except Exception as e:
            logging.error(f"An error occurred during parallel frame processing: {e}")
            return []

    all_keypoints = [kp for kp in all_keypoints if kp is not None]
    if not all_keypoints:
        logging.error("No valid keypoints extracted from any frame.")
        return []

    logging.info(f"Buffering {len(all_keypoints)} keypoint sets into sequences of length {sequence_length}...")
    sequences = []
    buffer = []
    for keypoint_frame in all_keypoints:
        buffer.append(keypoint_frame)
        if len(buffer) == sequence_length:
            sequences.append([frame_kp.tolist() for frame_kp in buffer])
            buffer = []
    logging.info(f"Finished buffering. Created {len(sequences)} sequences.")
    return sequences