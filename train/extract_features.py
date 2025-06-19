import os
import cv2
import glob
import time
import logging
import argparse
import numpy as np
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Hands solution globally
mp_hands = mp.solutions.hands

# Define constants based on MediaPipe Hands
NUM_HAND_LANDMARKS = 21
NUM_FEATURES_PER_HAND = NUM_HAND_LANDMARKS * 3 # x, y, z for each landmark
TOTAL_FEATURES_EXPECTED = NUM_FEATURES_PER_HAND * 2

# Thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

def check_gpu_availability():
    """Checks if OpenCV is built with CUDA and if a CUDA device is available."""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logging.info("CUDA (GPU) is available and detected by OpenCV.")
            return True
        else:
            logging.info("CUDA (GPU) is not available or not detected by OpenCV.")
            return False
    except AttributeError:
        logging.info("OpenCV is not built with CUDA support.")
        return False

def normalize_and_structure_keypoints(results: mp.solutions.hands.Hands.process) -> np.ndarray:
    """
    Normalizes hand landmarks relative to the wrist and structures them into a 126-feature array.
    This function precisely replicates the normalization logic from the frontend (page.tsx).
    """
    # Initialize placeholder arrays for left and right hands with 63 zeros each.
    user_left_hand_kps = np.zeros(NUM_FEATURES_PER_HAND, dtype=np.float32)
    user_right_hand_kps = np.zeros(NUM_FEATURES_PER_HAND, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            handedness_entry = results.multi_handedness[i]
            hand_label = handedness_entry.classification[0].label

            # Get wrist coordinates (landmark 0)
            wrist_landmark = landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist_landmark.x, wrist_landmark.y, wrist_landmark.z

            # Calculate wrist-relative coordinates for all 21 landmarks
            normalized_landmarks = []
            for lm in landmarks.landmark:
                normalized_landmarks.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])

            if hand_label == "Left":
                user_left_hand_kps = np.array(normalized_landmarks, dtype=np.float32)
            elif hand_label == "Right":
                user_right_hand_kps = np.array(normalized_landmarks, dtype=np.float32)

    # Combine left and right hand keypoints into a single 126-feature vector
    return np.concatenate([user_left_hand_kps, user_right_hand_kps])


def process_video(video_path: str, output_filepath: str):
    """
    Processes a single video file, extracts NORMALIZED keypoints for each frame,
    and saves the sequence of keypoints to a .npy file.
    """
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return False

    full_sequence = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB and process with MediaPipe to get hand landmark results
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Use the new normalization function to get the 126-feature vector for the current frame
            processed_keypoints = normalize_and_structure_keypoints(results)
            full_sequence.append(processed_keypoints)

    cap.release()

    if not full_sequence:
        logging.warning(f"No keypoint sequences extracted from {video_path}. Skipping save.")
        return False

    # Ensure output directory exists for the label
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    try:
        np.save(output_filepath, np.array(full_sequence, dtype=np.float32))
        logging.info(f"Successfully processed and saved {len(full_sequence)} frames to {output_filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving keypoints to {output_filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand keypoints from raw video files.")
    parser.add_argument('--input_dir', type=str, default='raw_videos',
                        help="Directory containing raw video files, structured as raw_videos/label/video.mp4.")
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help="Directory to save processed .npy keypoint files.")
    parser.add_argument('--check_gpu', action='store_true',
                        help="Check for OpenCV CUDA (GPU) availability.")

    args = parser.parse_args()

    if args.check_gpu:
        check_gpu_availability()

    os.makedirs(args.output_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)
    video_files.sort()

    if not video_files:
        logging.warning(f"No video files found in '{args.input_dir}'. Please ensure videos are recorded.")
        return

    logging.info(f"Found {len(video_files)} video files to process.")
    start_total_time = time.time()

    processed_count = 0
    failed_count = 0

    for video_file in video_files:
        parts = video_file.split(os.sep)

        if len(parts) < 2:
            logging.warning(f"Skipping malformed video path: {video_file}. Expected format 'raw_videos/label/video.mp4'.")
            failed_count += 1
            continue

        label = parts[-2]
        filename = os.path.basename(video_file)
        name_without_ext = os.path.splitext(filename)[0]

        output_filepath = os.path.join(args.output_dir, label, f"{name_without_ext}.npy")

        if process_video(video_file, output_filepath):
            processed_count += 1
        else:
            failed_count += 1

    end_total_time = time.time()
    logging.info(f"--- Processing complete ---")
    logging.info(f"Total videos processed: {processed_count}")
    logging.info(f"Total videos failed: {failed_count}")
    logging.info(f"Total time taken: {end_total_time - start_total_time:.2f} seconds.")

if __name__ == "__main__":
    main()