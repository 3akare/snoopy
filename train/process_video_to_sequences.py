import cv2
import os
import numpy as np
import mediapipe as mp
import logging
import sys

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

mp_holistic = mp.solutions.holistic

# Configuration constants
DATA_PATH = os.path.join("data")
SEQUENCE_LENGTH = 80

def mediapipe_detection(image, model):
    """Processes image using MediaPipe Holistic model."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results # Only return results, no need for image back if not drawing

def extract_keypoints(results):
    """
    Extracts keypoints from detected landmarks, focusing only on hands.
    Ensures consistency with the input size of your trained model (21*3 + 21*3 = 126 features).
    """
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    
    return np.concatenate([lh, rh]).astype(np.float32)


def process_video_and_save_sequences(video_path, action_name, data_path=DATA_PATH, sequence_length=SEQUENCE_LENGTH):
    """
    Processes a single video, extracts keypoints, segments into non-overlapping sequences, and saves them.
    """
    logging.info(f"Processing video: {video_path} for action: {action_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return

    action_dir = os.path.join(data_path, action_name)
    os.makedirs(action_dir, exist_ok=True)

    # Determine the next available sequence number
    existing_sequences = [
        int(d) for d in os.listdir(action_dir)
        if os.path.isdir(os.path.join(action_dir, d)) and d.isdigit()
    ]
    current_sequence_num = max(existing_sequences) + 1 if existing_sequences else 0
    logging.info(f"Starting sequence numbering from {current_sequence_num} for action '{action_name}'.")

    all_video_keypoints = []

    # Initialize Holistic model once for the entire video processing
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as holistic:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video or error
            
            # Ensure frame is not empty (e.g. from corrupt video)
            if frame is None or frame.size == 0:
                logging.warning(f"Skipping empty frame {frame_count} in {video_path}")
                frame_count += 1
                continue

            results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            ZERO_THRESHOLD_PER_FRAME = 0.9

            # Calculate the proportion of zero values in the current frame's keypoints
            percentage_zeros = np.sum(keypoints == 0) / keypoints.size

            # If the frame consists of too many zeros, log a warning and skip it.
            if percentage_zeros >= ZERO_THRESHOLD_PER_FRAME:
                logging.warning(
                    f"Skipping frame {frame_count} in '{video_path}' "
                    f"for action '{action_name}': {percentage_zeros*100:.2f}% of keypoints are zeros. "
                    "Hand detection likely failed."
                )
                frame_count += 1 # Ensure frame count is incremented for skipped frames
                continue # Skip adding this frame's keypoints and proceed to the next frame

            all_video_keypoints.append(keypoints)
            frame_count += 1
    cap.release()
    logging.info(f"Finished reading video. Extracted {len(all_video_keypoints)} frames of keypoints from {video_path}.")

    if len(all_video_keypoints) < sequence_length:
        logging.warning(f"Video too short ({len(all_video_keypoints)} frames) to create sequence of length {sequence_length} for {action_name}. Skipping.")
        return

    num_total_frames = len(all_video_keypoints)
    frames_processed = 0

    # Segment and save sequences (non-overlapping)
    while frames_processed + sequence_length <= num_total_frames:
        segment = all_video_keypoints[frames_processed : frames_processed + sequence_length]
        sequence_data_array = np.array(segment, dtype=np.float32) # Ensure float32

        sequence_dir = os.path.join(action_dir, str(current_sequence_num))
        os.makedirs(sequence_dir, exist_ok=True)

        save_path = os.path.join(sequence_dir, "sequence_data.npy")
        np.save(save_path, sequence_data_array)
        logging.info(f"Saved sequence {current_sequence_num} ({sequence_length} frames) for '{action_name}' to {save_path}")

        frames_processed += sequence_length # Move to the start of the next non-overlapping window
        current_sequence_num += 1
    
    logging.info(f"Finished segmenting and saving sequences from {video_path}. Saved up to sequence number {current_sequence_num - 1}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process video into gesture keypoint sequences.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('action_name', type=str, help='Name of the action in the video.')
    args = parser.parse_args()

    process_video_and_save_sequences(args.video_path, args.action_name)