import cv2
import os
import time
import argparse
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Define the default video resolution
RESOLUTION_WIDTH = 640
RESOLUTION_HEIGHT = 480
DEFAULT_DURATION = 10  # seconds
DEFAULT_NUM_RECORDINGS = 10 # Default number of recordings per gesture

# Initialize MediaPipe Hands solution globally
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def record_gestures(label: str, num_recordings: int = DEFAULT_NUM_RECORDINGS,
                   duration: int = DEFAULT_DURATION, output_dir: str = 'raw_videos'):
    """
    Records multiple videos of a sign gesture from the webcam, displaying MediaPipe detections live
    but saving clean videos.

    Args:
        label (str): The label (class) of the gesture (e.g., 'a', 'hello').
        num_recordings (int): The number of videos to record for this label.
        duration (int): The duration of each recording in seconds.
        output_dir (str): The base directory to save the raw videos.
    """
    # Create the directory for the label if it doesn't exist
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    logging.info(f"Ensured directory '{label_dir}' exists.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam. Please check if it's connected and not in use.")
        return

    # Set video resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)

    # Get frame rate and calculate video codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        logging.warning("Webcam reported 0 FPS. Defaulting to 30 FPS.")
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files

    logging.info(f"Webcam initialized: Resolution {RESOLUTION_WIDTH}x{RESOLUTION_HEIGHT}, FPS {fps}")

    # Initialize MediaPipe Hands model for processing frames outside the loop to avoid re-initialization
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
                cap.release()
                return

            logging.info(f"Recording gesture '{label}' video {i}/{num_recordings} for {duration} seconds...")
            logging.info(f"Saving clean video to: {output_filepath}")

            start_time = time.time()
            frames_recorded = 0
            
            # This flag will be set to True if 'q' is pressed, to break all loops
            global_quit_flag = False 

            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to grab frame. Exiting current recording.")
                    break

                # Flip the frame horizontally for a more intuitive display (optional, but common)
                frame = cv2.flip(frame, 1)

                # Create a copy of the frame for displaying MediaPipe detections
                display_frame = frame.copy()

                # Convert the frame to RGB for MediaPipe processing
                image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # Process the frame with MediaPipe Hands
                results = hands_model.process(image_rgb)
                image_rgb.flags.writeable = True

                # Draw hand landmarks on the display_frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the frame with MediaPipe detections
                cv2.imshow('Recording Gesture with Detections - Press Q to Quit', display_frame)

                # Write the ORIGINAL frame (without detections) to the output file
                out.write(frame)
                frames_recorded += 1

                # Check for 'q' press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("'q' pressed. Stopping all recordings and exiting.")
                    global_quit_flag = True
                    break
            
            # If 'q' was pressed, break out of the outer loop as well
            if global_quit_flag:
                break

            end_time = time.time()
            actual_duration = end_time - start_time

            # Release the video writer for the current video
            out.release()
            logging.info(f"Finished recording video {i}. Recorded {frames_recorded} frames in {actual_duration:.2f} seconds.")

    # Release webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam released and all windows closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record sign language gesture videos from webcam.")
    parser.add_argument('--label', type=str, required=True,
                        help="Label (class) of the gesture to record (e.g., 'a', 'hello').")
    parser.add_argument('--num_recordings', type=int, default=DEFAULT_NUM_RECORDINGS,
                        help=f"Number of videos to record for this label (default: {DEFAULT_NUM_RECORDINGS}).")
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION,
                        help=f"Duration of each recording in seconds (default: {DEFAULT_DURATION}).")
    parser.add_argument('--output_dir', type=str, default='raw_videos',
                        help="Base directory to save raw videos (default: 'raw_videos').")

    args = parser.parse_args()

    # Call the modified record_gesture function
    record_gesture(args.label, args.num_recordings, args.duration, args.output_dir)