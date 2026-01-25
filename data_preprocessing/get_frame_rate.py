from pathlib import Path
import cv2

INPUT_FILE = Path(
    "data/D_completely_xy_labeled_clips/PXL_20251124_223727362.TS_annotated_seg11.mp4"
)

# Open the video file
cap = cv2.VideoCapture(str(INPUT_FILE))

# Get the frame rate (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)

# Get other useful properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps > 0 else 0

print(f"Video: {INPUT_FILE.name}")
print(f"Frame Rate (FPS): {fps}")
print(f"Total Frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

cap.release()

"""
Video: PXL_20251124_223727362.TS_annotated_seg11.mp4
Frame Rate (FPS): 30.0
Total Frames: 30
Duration: 1.00 seconds
"""
