import cv2
import mediapipe as mp
import os
import time

def record_clean_video_with_gesture_preview(folder_name, num_videos=10, video_duration_sec=25, fps=20.0):
    """ Collect training data and saves to folder """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    total_frames_per_video = int(fps * video_duration_sec)
    print(f"Each video will contain exactly {total_frames_per_video} frames.")

    print(f"Recording will start in 3 seconds for folder: {folder_name}")
    time.sleep(3)

    for i in range(num_videos):
        video_filename = os.path.join(folder_name, f"{folder_name}_{i+1:02d}.mp4")
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

        print(f"Recording video {i+1}/{num_videos}: {video_filename}")
        frames_recorded_current_video = 0

        while frames_recorded_current_video < total_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame for video {i+1}. Exiting.")
                break

            frame_to_save = frame.copy()
            frame_for_display = cv2.flip(frame.copy(), 1)

            rgb_frame_for_mp = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame_for_mp)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame_for_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand Tracking (NOT SAVED) & Recording Clean Video', frame_for_display)

            out.write(frame_to_save)
            frames_recorded_current_video += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user.")
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return

        out.release()
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("All videos recorded successfully!" if i == num_videos - 1 else "Recording interrupted.")

if __name__ == "__main__":
    user_folder_name = input("Enter the name for the video folder: ")
    number_of_videos = 10
    duration_per_video = 25
    frames_per_second = 20.0

    record_clean_video_with_gesture_preview(user_folder_name, number_of_videos, duration_per_video, frames_per_second)