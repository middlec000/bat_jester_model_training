from time import time
import subprocess
import argparse

overall_time_start = time()

parser = argparse.ArgumentParser(description="Run all preprocessing steps")
parser.add_argument(
    "--run",
    nargs="+",
    default=["new"],
    help='Run mode: "all" to process all files, "new" to process only unprocessed files (default), or provide one or more substrings to process all files whose names contain any substring',
)
args = parser.parse_args()
run_arg = args.run
# Build args to forward to each preprocessing script
run_forward = ["--run"] + run_arg
print(f"Running all preprocessing steps (forwarding --run {run_arg})...\n")

for preprocessing_step_script in [
    "scripts/data_preprocessing_steps/1_clip_beginning_and_end.py",
    "scripts/data_preprocessing_steps/2_label_ball_xy_position.py",
    "scripts/data_preprocessing_steps/3_remove_xy_unlabeled_frames.py",
    "scripts/data_preprocessing_steps/4_label_juggle_timestamps.py",
    "scripts/data_preprocessing_steps/5_separate_audio_from_video.py",
    "scripts/data_preprocessing_steps/6_labeled_audio_plots.py",
]:
    script_start_time = time()
    print(f"Running {preprocessing_step_script}...")
    result = subprocess.run(
        ["uv", "run", "python", preprocessing_step_script] + run_forward
    )
    if result.returncode != 0:
        print(
            f"Error: {preprocessing_step_script} failed with return code {result.returncode}"
        )
        break
    print(
        f"Finished {preprocessing_step_script} in {time() - script_start_time:.2f} seconds.\n"
    )

overall_time_end = time()
print(
    f"Total time for all preprocessing steps: {overall_time_end - overall_time_start:.2f} seconds"
)


"""
Example:
uv run python scripts/run_all_data_labeling_steps.py --run all
uv run python scripts/run_all_data_labeling_steps.py --run new
uv run python scripts/run_all_data_labeling_steps.py --run PXL_20251202_230133137
"""
