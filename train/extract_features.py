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

# Initialize MediaPipe Hands solution globally for efficiency
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the number of expected landmarks per hand
NUM_HAND_LANDMARKS = 21
NUM_FEATURES_PER_HAND = NUM_HAND_LANDMARKS * 3 # x, y, z for each landmark

# Thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
# Frame quality threshold: percentage of non-zero keypoints required
MIN_NON_ZERO_KEYPOINTS_RATIO = 0.8 # At least 80% of detected keypoints must be non-zero

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

def extract_keypoints(frame: np.ndarray, hands_model: mp.solutions.hands.Hands):
    """
    Extracts hand keypoints from a single frame using MediaPipe.

    Args:
        frame (np.ndarray): The input image frame (BGR format).
        hands_model (mp.solutions.hands.Hands): Initialized MediaPipe Hands model.

    Returns:
        tuple: A tuple containing:
            - list: Normalized keypoints for the left hand (x, y, z for each landmark).
                    Returns an empty list if not detected or not fully detected.
            - list: Normalized keypoints for the right hand.
            - int: Number of hands detected in the frame (0, 1, or 2).
    """
    # Convert to RGB and set writable flag for performance
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands_model.process(image_rgb)

    # Set writable flag back (good practice, though not strictly needed if image_rgb isn't modified later)
    image_rgb.flags.writeable = True

    left_hand_keypoints = []
    right_hand_keypoints = []
    num_hands_detected = 0

    if results.multi_hand_landmarks:
        num_hands_detected = len(results.multi_hand_landmarks)
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness_info = results.multi_handedness[hand_idx].classification[0]
            score = handedness_info.score
            label = handedness_info.label # 'Left' or 'Right' (from the user's perspective)

            if score > MIN_HAND_DETECTION_CONFIDENCE:
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])

                # Validate if all expected landmarks are present and if keypoints are mostly non-zero
                if len(keypoints) == NUM_FEATURES_PER_HAND:
                    non_zero_count = np.count_nonzero(keypoints)
                    if non_zero_count / len(keypoints) >= MIN_NON_ZERO_KEYPOINTS_RATIO:
                        if label == 'Left':
                            left_hand_keypoints = keypoints
                        elif label == 'Right':
                            right_hand_keypoints = keypoints
                    else:
                        logging.debug(f"Skipping hand due to too many zero keypoints ({non_zero_count}/{len(keypoints)}).")
                else:
                    logging.debug(f"Skipping hand due to incorrect number of keypoints detected ({len(keypoints)} != {NUM_FEATURES_PER_HAND}).")

    return left_hand_keypoints, right_hand_keypoints, num_hands_detected


def process_video(video_path: str, output_filepath: str = 'processed_data'):
    """
    Processes a single video to extract hand keypoints.
    Frames are skipped if no hands are detected or if detected hands don't meet quality thresholds.

    Args:
        video_path (str): Path to the input video file.
        output_filepath (str): Path to save the extracted keypoints NumPy array.

    Returns:
        bool: True if keypoints were successfully extracted and saved, False otherwise.
    """
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return False

    all_frames_keypoints = []
    processed_frame_count = 0
    skipped_no_hand_frames = 0
    skipped_poor_quality_frames = 0
    
    with mp_hands.Hands(
        min_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        max_num_hands=2
    ) as hands_model:
        while True: # Loop until no more frames or error
            ret, frame = cap.read()
            if not ret:
                break # End of video or error

            processed_frame_count += 1
            left_kp, right_kp, num_hands = extract_keypoints(frame, hands_model)
            if num_hands == 0:
                skipped_no_hand_frames += 1
                logging.debug(f"Skipping frame {processed_frame_count} in {video_path}: No hands detected.")
                continue # Skip to the next frame

            # If neither hand passed the quality checks in extract_keypoints, then both left_kp and right_kp will be empty.
            if not left_kp and not right_kp:
                skipped_poor_quality_frames += 1
                logging.debug(f"Skipping frame {processed_frame_count} in {video_path}: Hands detected but keypoints failed quality check.")
                continue # Skip to the next frame

            # If we reach here, at least one hand with good quality keypoints was found.
            # Now, form the combined keypoints, padding with zeros if only one hand is visible.
            combined_keypoints = []
            if len(left_kp) == NUM_FEATURES_PER_HAND and len(right_kp) == NUM_FEATURES_PER_HAND:
                combined_keypoints = left_kp + right_kp
                logging.debug(f"Frame {processed_frame_count}: Both hands detected and valid.")
            elif len(left_kp) == NUM_FEATURES_PER_HAND:
                combined_keypoints = left_kp + [0.0] * NUM_FEATURES_PER_HAND
                logging.debug(f"Frame {processed_frame_count}: Only left hand detected and valid.")
            elif len(right_kp) == NUM_FEATURES_PER_HAND:
                combined_keypoints = [0.0] * NUM_FEATURES_PER_HAND + right_kp
                logging.debug(f"Frame {processed_frame_count}: Only right hand detected and valid.")
            
            # This 'else' should theoretically not be hit due to the skip logic above,
            # but as a safeguard, it prevents adding empty or invalid keypoints.
            else:
                logging.error(f"Logic error: Frame {processed_frame_count} passed skip conditions but has no valid keypoints. This should not happen.")
                skipped_poor_quality_frames += 1 # Count it as skipped
                continue # Skip the frame

            all_frames_keypoints.append(np.array(combined_keypoints, dtype=np.float32))
    cap.release()

    if not all_frames_keypoints:
        logging.warning(f"No valid frames with hand detection found in {video_path}. Skipping save.")
        return False

    # Convert list of arrays to a single NumPy array
    extracted_data = np.array(all_frames_keypoints, dtype=np.float32)

    # Final validation of extracted data
    expected_features = NUM_FEATURES_PER_HAND * 2 # 2 hands * 21 landmarks * 3 coords
    if extracted_data.size == 0 or extracted_data.shape[1] != expected_features:
        logging.error(f"Error: Invalid extracted data shape for {video_path}. Expected {expected_features} features, got {extracted_data.shape[1]}. Skipping save.")
        return False
    
    # Check for overall "emptiness" of the extracted data
    total_elements = extracted_data.size
    total_non_zeros = np.count_nonzero(extracted_data)
    if total_non_zeros / total_elements < MIN_NON_ZERO_KEYPOINTS_RATIO:
        logging.warning(f"Extracted data for {video_path} is mostly zeros ({total_non_zeros}/{total_elements} non-zero). Data might be poor. Still saving but investigate.")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    # Save the extracted data
    try:
        np.save(output_filepath, extracted_data)
        logging.info(f"Successfully extracted {extracted_data.shape[0]} frames with {extracted_data.shape[1]} features each from {video_path}.")
        logging.info(f"Skipped {skipped_no_hand_frames} frames due to no hand detection.")
        logging.info(f"Skipped {skipped_poor_quality_frames} frames due to poor hand keypoint quality.")
        logging.info(f"Saved keypoints to: {output_filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving keypoints to {output_filepath}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract hand keypoints from raw gesture videos using MediaPipe.")
    parser.add_argument('--input_dir', type=str, default='raw_videos',
                        help="Directory containing raw video files (e.g., 'raw_videos/a/a_01.mp4').")
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help="Directory to save processed keypoint NumPy arrays.")
    parser.add_argument('--check_gpu', action='store_true',
                        help="Flag to check for GPU availability using OpenCV's CUDA module.")

    args = parser.parse_args()

    if args.check_gpu:
        check_gpu_availability() # Logs GPU status

    # Find all mp4 files in the input directory and its subdirectories
    video_files = glob.glob(os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)

    if not video_files:
        logging.warning(f"No video files found in '{args.input_dir}'. Please ensure videos are recorded.")
        return

    logging.info(f"Found {len(video_files)} video files to process.")
    start_total_time = time.time()

    processed_count = 0
    failed_count = 0

    for video_file in video_files:
        # Extract label and index from the video file path
        parts = video_file.split(os.sep)
        
        # Ensure we can safely get the label (folder name)
        if len(parts) < 2:
            logging.warning(f"Skipping malformed video path: {video_file}. Expected format 'raw_videos/label/video.mp4'.")
            failed_count += 1
            continue

        label = parts[-2] # The folder name is the label
        filename = os.path.basename(video_file) # e.g., a_01.mp4
        name_without_ext = os.path.splitext(filename)[0] # e.g., a_01

        # Construct the output path
        output_filepath = os.path.join(args.output_dir, label, f"{name_without_ext}.npy")

        # Process the video
        if process_video(video_file, output_filepath):
            processed_count += 1
        else:
            failed_count += 1
    
    end_total_time = time.time()
    logging.info(f"--- Processing complete ---")
    logging.info(f"Total videos processed: {processed_count}")
    logging.info(f"Total videos failed: {failed_count}")
    logging.info(f"Total time taken: {end_total_time - start_total_time:.2f} seconds")


if __name__ == "__main__":
    main()