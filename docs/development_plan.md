# Data Collection
- [x] Collect videos from Colin
    - I have some, can take more
    - See Google Drive folder
- [ ] Collect videos from a wider user-base
    - Will do this after preprocessing pipeline is developed and we have the requirements on the videos

# Data Preprocessing
- [x] Load each video into Python
- [ ] Clip beginning and end of video so only video where juggling is happening remains
    - Use a speech model to identify where I say “start” and “stop” and trim before and after
- [x] Label x, y position of ball in each video frame using image model
- [ ] Remove video where no x, y position could be found and save remaining as separate video segments (with separate x, y label files)
- [x] Label ball juggle timestamps
    - Calculate vertical ball velocity from x, y positions
    - Add juggle timestamp label when ball switches from downward velocity to upward
- [ ] Check that labels match up with data by plotting audio wave and juggle labels

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
