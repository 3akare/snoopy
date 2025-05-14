#!/bin/bash
python3 process_video_to_sequences.py raw_videos/D.mov D
python3 process_video_to_sequences.py raw_videos/A.mov A
python3 train.py