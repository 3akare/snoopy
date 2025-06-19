import os
import cv2
import time
import logging
import argparse
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
RESOLUTION_WIDTH = 640
RESOLUTION_HEIGHT = 480
DEFAULT_DURATION = 10
DEFAULT_NUM_RECORDINGS = 10

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def record_gestures(label: str, num_recordings: int, duration: int, output_dir: str):
    """
    Records multiple videos of a sign gesture from the webcam.
    """
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    logging.info(f"Ensured directory '{label_dir}' exists.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        logging.warning("Webcam reported 0 FPS. Defaulting to 30 FPS.")
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    logging.info(f"Webcam initialized: Resolution {RESOLUTION_WIDTH}x{RESOLUTION_HEIGHT}, FPS {fps}")

    with mp_hands.Hands(
        min_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        max_num_hands=2
    ) as hands_model:
        for i in range(1, num_recordings + 1):
            output_filepath = os.path.join(label_dir, f"{label}_{i:02d}.mp4")
            out = cv2.VideoWriter(output_filepath, fourcc, fps, (RESOLUTION_WIDTH, RESOLUTION_HEIGHT))

            if not out.isOpened():
                logging.error(f"Error: Could not open video writer for {output_filepath}.")
                break

            logging.info(f"Recording gesture '{label}' video {i}/{num_recordings} for {duration} seconds...")
            logging.info(f"Saving to: {output_filepath}")

            start_time = time.time()
            global_quit_flag = False

            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to grab frame. Exiting recording.")
                    break

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = hands_model.process(image_rgb)
                image_rgb.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('Recording Gesture - Press Q to Quit', display_frame)
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("'q' pressed. Stopping all recordings.")
                    global_quit_flag = True
                    break

            out.release()
            logging.info(f"Finished recording video {i}.")

            if global_quit_flag:
                break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam released and all windows closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record sign language gesture videos from webcam.")
    parser.add_argument('--label', type=str, required=True, help="Label (class) of the gesture to record.")
    parser.add_argument('--num_recordings', type=int, default=DEFAULT_NUM_RECORDINGS, help="Number of videos to record for this label.")
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION, help="Duration of each recording in seconds.")
    parser.add_argument('--output_dir', type=str, default='raw_videos', help="Base directory to save raw videos.")
    args = parser.parse_args()

    record_gestures(args.label, args.num_recordings, args.duration, args.output_dir)