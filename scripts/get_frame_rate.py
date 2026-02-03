from pathlib import Path
import sys

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import utils

INPUT_FILE = Path(
    "/home/colin/data/bat_jester_model_training/D_completely_xy_labeled_clips/PXL_20251124_223727362.TS_seg1.mp4"
)
INPUT_FILE = Path(
    "/home/colin/data/bat_jester_model_training/D_completely_xy_labeled_clips/PXL_20251124_223727362.TS_seg1_annotated.mp4"
)

fps = utils.get_video_fps(INPUT_FILE)

print(f"Video: {INPUT_FILE.name}")
print(f"Frame Rate (FPS): {fps}")


"""
Video: PXL_20251124_223727362.TS_seg1.mp4
Frame Rate (FPS): 30.0

Video: PXL_20251124_223727362.TS_seg1_annotated.mp4
Frame Rate (FPS): 30.0
"""
