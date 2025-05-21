import os
from subprocess import call
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
    ]
)

# Configuration
RAW_VIDEOS_DIR = "raw_videos"
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.webm', '.avi') # Add all supported extensions

logging.info(f"Starting batch processing of videos in '{RAW_VIDEOS_DIR}'...")

# Walk through the directory tree
for root, dirs, files in os.walk(RAW_VIDEOS_DIR):
    for file_name in files:
        # Check if the file has a supported video extension
        if file_name.lower().endswith(VIDEO_EXTENSIONS):
            video_full_path = os.path.join(root, file_name)

            # Extract the label (gesture name) from the parent directory name
            # e.g., for raw_videos/A/A_01.mp4, root will be "raw_videos/A", so os.path.basename(root) is "A"
            label = os.path.basename(root).upper()

            if not label: # Handle case where video is directly in raw_videos if that's ever possible
                logging.warning(f"Could not determine label for {video_full_path}. Skipping.")
                continue

            logging.info(f"Processing video: '{video_full_path}' with detected label: '{label}'")
            
            try:
                # Call the process_video_to_sequences.py script
                # Ensure process_video_to_sequences.py is in the same directory as this script,
                # or adjust the path accordingly.
                command = ["python", "process_video_to_sequences.py", video_full_path, label]
                result = call(command)

                if result == 0:
                    logging.info(f"Successfully processed: {video_full_path}")
                else:
                    logging.error(f"Failed to process: {video_full_path}. Exit code: {result}")
            except Exception as e:
                logging.error(f"An error occurred while calling process_video_to_sequences.py for {video_full_path}: {e}", exc_info=True)

logging.info("Batch video processing completed.")