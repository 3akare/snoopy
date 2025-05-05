import cv2
import os
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Constants
DATA_PATH = os.path.join("data")
ACTIONS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    "Name",
    "Learn",
    "Restroom",
    "No",
    "What",
    "Sign",
    "Where",
    "Sister",
    "Nice",
    "Not",
    "Classroom",
    "Girl-friend",
    "You",
    "Student",
    "Buy",
    "Brother",
    "Meet",
    "Teacher",
    "Food",
    "Have"
]
NUM_SEQUENCES = 30  # Number of videos per action
SEQUENCE_LENGTH = 30  # Frames per video

# Create directories
for action in ACTIONS:
    for sequence in range(NUM_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

def mediapipe_detection(image, model):
    """Processes image using MediaPipe Holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def draw_landmarks(image, results):
    """Draws facial, pose, and hand landmarks."""
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    """Extracts keypoints from detected landmarks."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                    ).flatten() if results.face_landmarks else np.zeros(468 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                  ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                  ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Open Webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        print(f"\nGet ready to record '{action.upper()}'. Press 'q' anytime to quit.")
        time.sleep(2)  # Give user time to prepare

        for sequence in range(NUM_SEQUENCES):
            print(f"\nRecording Sequence {sequence + 1}/{NUM_SEQUENCES} for '{action}'")
            time.sleep(2)  # Short break between sequences

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not capture frame.")
                    continue

                # Process frame
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                # Display instructions
                cv2.putText(image, f"Recording '{action.upper()}'", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Sequence {sequence + 1}/{NUM_SEQUENCES}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
                cv2.putText(image, f"Frame {frame_num + 1}/{SEQUENCE_LENGTH}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2, cv2.LINE_AA)

                # Show webcam feed
                cv2.imshow("Data Collection", image)

                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit requested. Cleaning up.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                # Extract and save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

cap.release()
cv2.destroyAllWindows()

