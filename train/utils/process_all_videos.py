import os
from subprocess import call
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
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.webm', '.avi')
EXCLUDED_LABELS = {"C", "D", "L", "R", "U", "V", "W", "X", "Y", "Z", "NAME"}  # Directories to skip

logging.info(f"Starting batch processing of videos in '{RAW_VIDEOS_DIR}'...")

for root, dirs, files in os.walk(RAW_VIDEOS_DIR):
    label = os.path.basename(root).upper()
    if label in EXCLUDED_LABELS:
        logging.info(f"Skipping directory '{root}' with excluded label '{label}'")
        continue

    for file_name in files:
        if file_name.lower().endswith(VIDEO_EXTENSIONS):
            video_full_path = os.path.join(root, file_name)

            if not label:
                logging.warning(f"Could not determine label for {video_full_path}. Skipping.")
                continue
            logging.info(f"Processing video: '{video_full_path}' with detected label: '{label}'")

            try:
                command = ["python", "process_video_to_sequences.py", video_full_path, label]
                result = call(command)
                if result == 0:
                    logging.info(f"Successfully processed: {video_full_path}")
                else:
                    logging.error(f"Failed to process: {video_full_path}. Exit code: {result}")
            except Exception as e:
                logging.error(f"An error occurred while calling process_video_to_sequences.py for {video_full_path}: {e}", exc_info=True)
logging.info("Batch video processing completed.")
