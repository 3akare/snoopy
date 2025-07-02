import os
import cv2
import time
import logging
import argparse
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe Solution Objects
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuration
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720
DEFAULT_DURATION = 10
DEFAULT_NUM_RECORDINGS = 10

def record_gestures(label: str, num_recordings: int, duration: int, output_dir: str):
    """
    Records videos of a sign gesture, displaying real-time Pose and Hand landmarks.
    """
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    logging.info(f"Output directory: '{label_dir}'")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    logging.info(f"Webcam initialized: {int(cap.get(3))}x{int(cap.get(4))} @ {fps} FPS")

    pose_style = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
    left_hand_style = mp_drawing.DrawingSpec(color=(252, 12, 125), thickness=2, circle_radius=2)
    right_hand_style = mp_drawing.DrawingSpec(color=(12, 252, 220), thickness=2, circle_radius=2)

    with mp_hands.Hands(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, max_num_hands=2) as hands, \
         mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:

        for i in range(1, num_recordings + 1):
            output_filepath = os.path.join(label_dir, f"{label}_{i:02d}.mp4")
            out = cv2.VideoWriter(output_filepath, fourcc, fps, (RESOLUTION_WIDTH, RESOLUTION_HEIGHT))

            logging.info(f"Recording gesture '{label}', video {i}/{num_recordings}...")
            start_time = time.time()
            quit_flag = False

            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results_hands = hands.process(image_rgb)
                results_pose = pose.process(image_rgb)

                mp_drawing.draw_landmarks(display_frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=pose_style, connection_drawing_spec=pose_style)

                if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
                    for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        hand_label = results_hands.multi_handedness[idx].classification[0].label
                        style = left_hand_style if hand_label == "Left" else right_hand_style
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, style, mp_drawing_styles.get_default_hand_connections_style())

                cv2.imshow('Real-time Gesture Recording | Press [Q] to Quit', display_frame)
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    quit_flag = True
                    break
            out.release()
            logging.info(f"Saved clean video to: {output_filepath}")

            if quit_flag: break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Recording session finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record gesture videos with real-time landmark visualization.")
    parser.add_argument('--label', type=str, required=True, help="Label of the gesture to record.")
    parser.add_argument('--num_recordings', type=int, default=DEFAULT_NUM_RECORDINGS)
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION)
    parser.add_argument('--output_dir', type=str, default='raw_videos')
    args = parser.parse_args()

    record_gestures(args.label, args.num_recordings, args.duration, args.output_dir)