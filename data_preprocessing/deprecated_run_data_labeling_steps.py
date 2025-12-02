from pathlib import Path
from time import time
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from video_labeler import SoccerJuggleVideoLabeler

start_time = time()

raw_video_dir = Path("data/videos_to_process")
labeled_video_dir = Path("data/labeled_videos")
juggle_labels_dir = Path("data/juggle_labels")
audio_files_dir = Path("data/audio_files")
completed_videos_dir = Path("data/completed_videos")


for file_path in raw_video_dir.glob("*.mp4"):
    print(f"Processing video: {file_path}")

    # Initialize labeler
    labeler = SoccerJuggleVideoLabeler(video_path=file_path, confidence_threshold=0.001)
    # Process video (set visualize=True for visual feedback)
    labeler.process_video(
        visualize=False,
        output_path=labeled_video_dir / (file_path.stem + "_annotated.mp4"),
    )
    # Export labels
    labels = labeler.export_labels(
        juggle_labels_dir / (file_path.stem + "_labels.json")
    )
    print(f"Total juggles detected: {labels['total_juggles']}")

    # Extract audio from video
    audio_output_path = audio_files_dir / (file_path.stem + ".wav")
    print(f"Extracting audio to {audio_output_path}")
    video_clip = VideoFileClip(str(file_path))
    video_clip.audio.write_audiofile(str(audio_output_path), logger=None)
    video_clip.close()
    print(f"Audio extracted successfully to {audio_output_path}")

    # Move original video to completed_videos directory
    destination = completed_videos_dir / file_path.name
    shutil.move(str(file_path), str(destination))
    print(f"Moved {file_path.name} to {completed_videos_dir}")

end_time = time()

# Print summary
print(f"Labeling completed in {end_time - start_time:.2f} seconds")
