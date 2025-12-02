from pathlib import Path
from time import time
import pandas as pd
from video_labeler import SoccerJuggleVideoLabeler


INPUT_PATH = Path("data/1_clipped_videos")
OUTPUT_PATH = Path("data/2_ball_xy_positions")


def label_xy_positions():
    """
    Process all videos in the input directory and save:
    - Annotated videos with ball positions overlaid
    - Parquet files with frame-by-frame ball positions (Frame, x, y)
    """
    start_time = time()

    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Get all MP4 files from input directory
    video_files = list(INPUT_PATH.glob("*.mp4"))

    if not video_files:
        print(f"No video files found in {INPUT_PATH}")
        return

    print(f"Found {len(video_files)} video(s) to process")

    for file_path in video_files:
        print(f"\n{'=' * 60}")
        print(f"Processing video: {file_path.name}")
        print(f"{'=' * 60}")

        # Initialize labeler with low confidence threshold for ball detection
        labeler = SoccerJuggleVideoLabeler(
            video_path=str(file_path), confidence_threshold=0.001
        )

        # Process video and save annotated version
        annotated_output_path = OUTPUT_PATH / (file_path.stem + "_annotated.mp4")
        labeler.process_video(visualize=False, output_path=str(annotated_output_path))

        # Convert ball positions to DataFrame
        # ball_positions format: [(frame_num, x, y, confidence), ...]
        if labeler.ball_positions:
            df = pd.DataFrame(
                labeler.ball_positions, columns=["Frame", "x", "y", "confidence"]
            )

            # Save as parquet (without confidence column as per spec)
            parquet_output_path = OUTPUT_PATH / (
                file_path.stem + "_xy_frame_labels.parquet"
            )
            df[["Frame", "x", "y"]].to_parquet(parquet_output_path, index=False)

            print("\nSummary:")
            print(f"  - Frames processed: {df['Frame'].max() + 1}")
            print(f"  - Frames with ball detected: {len(df)}")
            print(f"  - Detection rate: {len(df) / (df['Frame'].max() + 1) * 100:.1f}%")
            print(f"  - Annotated video saved to: {annotated_output_path}")
            print(f"  - Labels saved to: {parquet_output_path}")
        else:
            print(f"Warning: No ball positions detected in {file_path.name}")

    end_time = time()

    # Print final summary
    print(f"\n{'=' * 60}")
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    label_xy_positions()
