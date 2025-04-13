import cv2
import numpy as np
import mediapipe as mp

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in (results.pose_landmarks.landmark if results.pose_landmarks else [])]).flatten() or np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in (results.face_landmarks.landmark if results.face_landmarks else [])]).flatten() or np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in (results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])]).flatten() or np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in (results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])]).flatten() or np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def process_video(sequence_length, holistic, video_path):
    cap = cv2.VideoCapture(video_path)
    sequences = []
    buffer = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        _, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)
        if len(buffer) == sequence_length:
            sequences.append(buffer.copy())  # Append the current sequence
            buffer.clear()  # Clear buffer for the next sequence
    cap.release()
    return sequences
