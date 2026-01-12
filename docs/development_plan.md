# Data Collection
- [x] Collect videos from Colin
    - I have some, can take more
    - See Google Drive folder
- [ ] Collect videos from a wider user-base
    - Will do this after preprocessing pipeline is developed and we have the requirements on the videos

# Data Preprocessing
- [x] (Step 1) Clip beginning and end of video so only video where juggling is happening remains
    - Use a speech model to identify where I say “start” and “stop” and trim before and after
    - Skip over (do not save) videos where either "start" or "stop" was not detected
    - Load from `data/0_raw_videos`
        - Original `.mp4` video files
    - Save to `/data/1_clipped_videos`
        - Clipped `.mp4` video files
    - Save to `/data/0_to_1_logs`
        - Processing logs
- [x] (Step 2) Label x, y position of ball in each video frame using image model
    - Load from `/data/1_clipped_videos`
        - Clipped `.mp4` video files
        - `.mp4` video files with xy position overlaid over every frame
    - Save to `/data/2_ball_xy_positions`
        - `.parquet` of xy labels: Frame, x, y
        - Frames where xy position could not be detected have NULL for x and y
- [ ] (Step 3) Remove video segments where no x, y position could be found and save remaining as separate video segments (with separate x, y label files)
    - Require segments to have a minimum duration of 1 second = 30 frames (@29.78150102817087 fps)
    - Load from `/data/1_clipped_videos`
        - Clipped `.mp4` video files
    - Load from `/data/2_ball_xy_positions`
        - `.parquet` of xy labels: Frame, x, y
    - Save to `/data/3_completely_xy_labeled_clips`
        - `.mp4` video files where every frame has an xy label
        - `.mp4` video files with xy position overlaid over every frame
        - `.parquet` of xy labels: Frame, x, y
- [ ] (Step 4) Label ball juggle timestamps
    - Not done but in [deprecated_run_data_labeling_steps.py](../data_preprocessing/deprecated_run_data_labeling_steps.py)
    - Calculate vertical ball velocity from x, y positions
    - Add juggle timestamp label when ball switches from downward velocity to upward
    - Load from `/data/3_completely_xy_labeled_clips`
        - `.mp4` video files where every frame has an xy label
        - `.parquet` of xy labels: Frame, x, y
    - Save to `/data/4_juggle_labels`
        - `.mp4` video files where every frame has an xy label
        - `.json` with timestamps and length of clip
- [ ] (Step 5) Separate the audio from the video
    - Not done but in [deprecated_run_data_labeling_steps.py](../data_preprocessing/deprecated_run_data_labeling_steps.py)
    - Load from `/data/4_juggle_labels`
        - `.mp4` video files where every frame has an xy label
    - Save to `/dev/5_audio/`
        - `.wav` files
- [ ] (Step 6) Check that labels match up with data by plotting audio wave and juggle labels
    - Load from `/data/4_juggle_labels`
        - `.json` with timestamps and length of clip
    - Load from `/dev/5_audio/`
        - `.wav` files
    - Save to `/data/6_audio_plots_with_labels`

# Model Selection
- [ ] Research models that can somehow predict number of juggles on a large audio sample
- [ ] Research binary classification models that can predict on windowed (chunked) audio
- [ ] Research if there are other ways to approach this problem

# Model Training
- [ ] Extract additional features from the input data.
    - [ ] Refine Wavelet approach.
- [ ] Add speech recognition to "Start" and "Stop" the juggle counter.

# Model Evaluation
- Do we need to evaluate the speech recognition part?
