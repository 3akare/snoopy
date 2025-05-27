import cv2
import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

RAW_VIDEOS_DIR = "raw_videos"
LOG_FILE_PATH = os.path.join("logs", "video_frame_check.log")

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.webm', '.avi')

def check_video_frames(video_path):
    """Checks a single video file for its total frames and read success."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file: {video_path}")
        return video_path, 0, 0, False, "Failed to open video"

    total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_successfully_read = 0
    warning_message = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            if frames_successfully_read < total_frames_reported:
                warning_message = f"WARNING: Only {frames_successfully_read} frames successfully read (expected {total_frames_reported})"
            break
        frames_successfully_read += 1
    
    cap.release()
    return (
        video_path,
        total_frames_reported,
        frames_successfully_read,
        frames_successfully_read >= total_frames_reported,
        warning_message
    )

if __name__ == "__main__":
    logging.info(f"Starting video frame count check in '{RAW_VIDEOS_DIR}'...")

    with open(LOG_FILE_PATH, "w") as f_log:
        f_log.write("--- Video Frame Check Results ---\n")
        for root, dirs, files in os.walk(RAW_VIDEOS_DIR):
            for file_name in files:
                if file_name.lower().endswith(VIDEO_EXTENSIONS):
                    video_full_path = os.path.join(root, file_name)
                    
                    path, total, read, success, warning = check_video_frames(video_full_path)
                    log_entry = f"Video: {path}, Reported Frames: {total}, Read Frames: {read}"
                    if warning:
                        log_entry += f", {warning}"
                        logging.warning(log_entry)
                    elif read == 0:
                        log_entry += ", ERROR: No frames could be read"
                        logging.error(log_entry)
                    else:
                        logging.info(log_entry)
                    
                    f_log.write(log_entry + "\n")

    logging.info(f"Video frame check completed. Results logged to '{LOG_FILE_PATH}'.")