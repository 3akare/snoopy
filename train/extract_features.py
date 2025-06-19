import os
import cv2
import glob
import time
import logging
import argparse
import numpy as np
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe Solution Objects
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Define Landmark Indices and Feature Dimensions
POSE_LANDMARKS_TO_EXTRACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

NUM_FACE_FEATURES = 468 * 3  # 468 landmarks with x, y, z
NUM_POSE_FEATURES = len(POSE_LANDMARKS_TO_EXTRACT) * 3
NUM_HAND_FEATURES = 21 * 3

TOTAL_FEATURES = NUM_FACE_FEATURES + NUM_POSE_FEATURES + (NUM_HAND_FEATURES * 2)

# MediaPipe Model Configuration
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

def extract_combined_features(results_face, results_hands, results_pose) -> np.ndarray:
    """
    Extracts, normalizes, and combines features from Face, Pose, and Hands models.
    """
    face_kps = np.zeros(NUM_FACE_FEATURES, dtype=np.float32)
    pose_kps = np.zeros(NUM_POSE_FEATURES, dtype=np.float32)
    left_hand_kps = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)
    right_hand_kps = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)

    # Process Face Landmarks
    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark
        # Normalize relative to landmark 1 (nose bridge)
        ref_point = face_landmarks[1]
        ref_x, ref_y, ref_z = ref_point.x, ref_point.y, ref_point.z
        face_coords = []
        for lm in face_landmarks:
            face_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
        face_kps = np.array(face_coords, dtype=np.float32)

    # Process Pose Landmarks
    if results_pose.pose_landmarks:
        pose_landmarks = results_pose.pose_landmarks.landmark
        ref_point = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
        ref_x, ref_y, ref_z = ref_point.x, ref_point.y, ref_point.z
        
        pose_coords = []
        for i in POSE_LANDMARKS_TO_EXTRACT:
            lm = pose_landmarks[i]
            pose_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
        pose_kps = np.array(pose_coords, dtype=np.float32)

    # Process Hand Landmarks
    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for i, landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[i].classification[0].label
            wrist_lm = landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist_lm.x, wrist_lm.y, wrist_lm.z
        
            hand_coords = []
            for lm in landmarks.landmark:
                hand_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            
            if handedness == "Left":
                left_hand_kps = np.array(hand_coords, dtype=np.float32)
            elif handedness == "Right":
                right_hand_kps = np.array(hand_coords, dtype=np.float32)

    return np.concatenate([face_kps, pose_kps, left_hand_kps, right_hand_kps])

def process_video(video_path: str, output_filepath: str):
    """
    Processes a video, extracts combined keypoints from all three models, and saves the sequence.
    """
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return False

    full_sequence = []
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as face_mesh, \
         mp_hands.Hands(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, max_num_hands=2) as hands, \
         mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(rgb_frame)
            results_hands = hands.process(rgb_frame)
            results_pose = pose.process(rgb_frame)
            
            combined_keypoints = extract_combined_features(results_face, results_hands, results_pose)
            full_sequence.append(combined_keypoints)
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
    parser = argparse.ArgumentParser(description="Extract Face, Pose, and Hand keypoints from videos.")
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
    logging.info(f"Final feature dimension per frame: {TOTAL_FEATURES}")

if __name__ == "__main__":
    main()