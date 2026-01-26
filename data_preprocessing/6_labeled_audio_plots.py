from pathlib import Path
from time import time
import sys

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent))

from bat_logging import logging_setup
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

LABELS_DIR = Path("data/E_juggle_labels")
AUDIO_DIR = Path("data/F_audio_extracted_from_videos")
OUTPUT_DIR = Path("data/G_audio_plots_with_labels")
LOGGING_DIR = Path("data/6_logs")

logger = logging_setup.get_processing_logger(LOGGING_DIR)

input_files = list(AUDIO_DIR.glob("*.wav"))
processed_files = list(OUTPUT_DIR.glob("*.png"))
processed_files = list()
unprocessed_files = [
    f
    for f in input_files
    if (OUTPUT_DIR / f.with_suffix(".png").name) not in processed_files
]

for audio_filename in unprocessed_files:
    label_file = LABELS_DIR / (
        audio_filename.stem + "_xy_frame_labels_juggle_timestamps.txt"
    )
    output_file = OUTPUT_DIR / audio_filename.with_suffix(".png").name
    start_time = time()
    try:
        # Load audio data
        audio_data, sr = librosa.load(str(audio_filename), sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sr)

        # Load timestamp labels
        if label_file.exists():
            with open(label_file, "r") as f:
                timestamps = [
                    float(line.strip()) for line in f.readlines() if line.strip()
                ]
        else:
            timestamps = []

        # Create figure and plot waveform
        fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

        # Plot audio waveform
        time_axis = np.linspace(0, duration, len(audio_data))
        ax.plot(
            time_axis, audio_data, linewidth=0.5, color="blue", alpha=0.7, label="Audio"
        )

        # Overlay timestamp labels as vertical lines
        for timestamp in timestamps:
            if 0 <= timestamp <= duration:
                ax.axvline(
                    x=timestamp, color="red", linewidth=1.5, alpha=0.6, linestyle="--"
                )

        # Add legend only if there are timestamps
        if timestamps:
            ax.axvline(
                x=timestamps[0],
                color="red",
                linewidth=1.5,
                alpha=0.6,
                linestyle="--",
                label="Juggle Catches",
            )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Audio: {audio_filename.stem}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.linspace(0, duration, 10))

        # Save plot
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), bbox_inches="tight", dpi=100)
        plt.close(fig)

        elapsed_time = time() - start_time
        logger.info(
            f"Generated plot for {audio_filename.name} in {elapsed_time:.2f} seconds and saved to {output_file.name}"
        )

    except Exception as e:
        logger.error(f"Failed to generate plot for {audio_filename.name}: {str(e)}")
