import cv2
import numpy as np
import mediapipe as mp

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

def process_video(sequence_length, holistic, video_path):
    cap = cv2.VideoCapture(video_path)
    sequences = []
    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)

        if len(buffer) == sequence_length:
            sequences.append(list(buffer))
            buffer = []

    cap.release()
    return sequences
