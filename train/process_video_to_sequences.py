import cv2
import os
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DATA_PATH = os.path.join("data")
SEQUENCE_LENGTH = 80

def mediapipe_detection(image, model):
    """Processes image using MediaPipe Holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    """Extracts keypoints from detected landmarks."""
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
    #                 ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
    #                 ).flatten() if results.face_landmarks else np.zeros(468 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                  ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                  ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    # return np.concatenate([pose, face, lh, rh])
    return np.concatenate([lh, rh]) # focus on only hand gestures... for now

def process_video_and_save_sequences(video_path, action_name, data_path=DATA_PATH, sequence_length=SEQUENCE_LENGTH):
    """
    Processes a single video, extracts keypoints, segments into sequences, and saves.
    """
    print(f"Processing video: {video_path} for action: {action_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    action_dir = os.path.join(data_path, action_name)
    os.makedirs(action_dir, exist_ok=True) # Ensure action directory exists
    existing_sequences = [int(d) for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d)) and d.isdigit()]
    current_sequence_num = max(existing_sequences) + 1 if existing_sequences else 0
    print(f"Starting sequence numbering from {current_sequence_num}")

    all_video_keypoints = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            all_video_keypoints.append(keypoints)
            frame_count += 1
    cap.release()
    print(f"Finished reading video. Extracted {len(all_video_keypoints)} frames of keypoints.")

    if len(all_video_keypoints) < sequence_length:
        print(f"Warning: Video too short ({len(all_video_keypoints)} frames) to create sequence of length {sequence_length}. Skipping.")
        return
    num_total_frames = len(all_video_keypoints)
    frames_processed = 0

    while frames_processed + sequence_length <= num_total_frames:
        # Extract a segment of keypoints
        segment = all_video_keypoints[frames_processed : frames_processed + sequence_length]

        # Convert segment (list of keypoints) to NumPy array
        sequence_data_array = np.array(segment)

        # Create directory for the current sequence number
        sequence_dir = os.path.join(action_dir, str(current_sequence_num))
        os.makedirs(sequence_dir, exist_ok=True)

        # Define save path and save the sequence data
        save_path = os.path.join(sequence_dir, "sequence_data.npy")
        np.save(save_path, sequence_data_array)
        print(f"Saved sequence {current_sequence_num} ({sequence_length} frames) for '{action_name}' to {save_path}")

        # Move to the start of the next non-overlapping window
        frames_processed += sequence_length
        current_sequence_num += 1
    print(f"Finished segmenting and saving sequences from {video_path}. Saved up to sequence number {current_sequence_num - 1}.")

# python process_video_to_sequences.py raw_videos/D_01.mp4 D
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process video into gesture keypoint sequences.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('action_name', type=str, help='Name of the action in the video.')
    # Optional: add arguments for sequence_length or data_path if you want to override constants
    args = parser.parse_args()

    # Ensure action name directory is created (done inside the function too, but harmless here)
    os.makedirs(os.path.join(DATA_PATH, args.action_name), exist_ok=True)
    process_video_and_save_sequences(args.video_path, args.action_name)
