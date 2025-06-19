import os
import cv2
import glob
import time
import logging
import argparse
import numpy as np
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mp_hands = mp.solutions.hands
NUM_FEATURES_PER_HAND = 21 * 3
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

def normalize_and_structure_keypoints(results: mp.solutions.hands.Hands.process) -> np.ndarray:
    """
    Normalizes hand landmarks relative to the wrist and structures them into a 126-feature array.
    """
    user_left_hand_kps = np.zeros(NUM_FEATURES_PER_HAND, dtype=np.float32)
    user_right_hand_kps = np.zeros(NUM_FEATURES_PER_HAND, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            handedness_entry = results.multi_handedness[i]
            hand_label = handedness_entry.classification[0].label
            wrist_landmark = landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist_landmark.x, wrist_landmark.y, wrist_landmark.z

            normalized_landmarks = []
            for lm in landmarks.landmark:
                normalized_landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

            if hand_label == "Left":
                user_left_hand_kps = np.array(normalized_landmarks, dtype=np.float32)
            elif hand_label == "Right":
                user_right_hand_kps = np.array(normalized_landmarks, dtype=np.float32)

    return np.concatenate([user_left_hand_kps, user_right_hand_kps])

def process_video(video_path: str, output_filepath: str):
    """
    Processes a video, extracts normalized keypoints, and saves the sequence to a .npy file.
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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            processed_keypoints = normalize_and_structure_keypoints(results)
            full_sequence.append(processed_keypoints)
    cap.release()

    if not full_sequence:
        logging.warning(f"No keypoints extracted from {video_path}. Skipping.")
        return False

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    try:
        np.save(output_filepath, np.array(full_sequence, dtype=np.float32))
        logging.info(f"Saved {len(full_sequence)} frames to {output_filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving keypoints to {output_filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand keypoints from videos.")
    parser.add_argument('--input_dir', type=str, default='raw_videos', help="Directory with raw videos.")
    parser.add_argument('--output_dir', type=str, default='processed_data', help="Directory to save .npy files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)
    video_files.sort()

    if not video_files:
        logging.warning(f"No video files found in '{args.input_dir}'.")
        return

    logging.info(f"Found {len(video_files)} videos to process.")
    start_time = time.time()
    processed_count, failed_count = 0, 0

    for video_file in video_files:
        label = video_file.split(os.sep)[-2]
        filename = os.path.basename(video_file)
        name_without_ext = os.path.splitext(filename)[0]
        output_filepath = os.path.join(args.output_dir, label, f"{name_without_ext}.npy")

        if process_video(video_file, output_filepath):
            processed_count += 1
        else:
            failed_count += 1

    end_time = time.time()
    logging.info("--- Processing Complete ---")
    logging.info(f"Videos processed: {processed_count}")
    logging.info(f"Videos failed: {failed_count}")
    logging.info(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()