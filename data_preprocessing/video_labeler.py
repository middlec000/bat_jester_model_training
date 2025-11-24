import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import deque


class SoccerJuggleVideoLabeler:
    def __init__(self, video_path, model_path="yolov8n.pt", confidence_threshold=0.25):
        """
        Initialize the juggle labeler

        Args:
            video_path: Path to the video file
            model_path: Path to YOLO model (default uses nano pretrained model)
            confidence_threshold: Minimum confidence for ball detection (default 0.25)
        """
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.confidence_threshold = confidence_threshold

        # Parameters for juggle detection
        self.velocity_threshold = 5  # pixels per frame
        self.min_frames_between_juggles = int(self.fps * 0.2)  # 200ms minimum

        # Storage
        self.ball_positions = []  # [(frame_num, x, y, confidence)]
        self.juggle_frames = []  # Frame numbers where juggles occur

    def detect_ball(self, frame):
        """
        Detect soccer ball in a frame
        Returns: (x, y, confidence) or None if not detected
        """
        # Pass confidence threshold to YOLO
        results = self.model(
            frame, classes=[32], conf=self.confidence_threshold, verbose=False
        )

        if len(results[0].boxes) > 0:
            # Get the detection with highest confidence
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()

            # Calculate center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            return center_x, center_y, confidence

        return None

    def calculate_velocity(self, positions, window_size=3):
        """
        Calculate vertical velocity from position history
        Uses a sliding window to smooth out noise
        """
        if len(positions) < window_size + 1:
            return None

        # Get recent positions
        recent = positions[-window_size - 1 :]

        # Calculate average velocity over window
        velocities = []
        for i in range(len(recent) - 1):
            dy = recent[i + 1][2] - recent[i][2]  # y position difference
            velocities.append(dy)

        return np.mean(velocities)

    def detect_juggles(self):
        """
        Detect juggle events based on velocity changes
        A juggle occurs when vertical velocity changes from downward to upward
        """
        velocity_history = deque(maxlen=5)
        last_juggle_frame = -999

        for i in range(1, len(self.ball_positions)):
            frame_num, x, y, conf = self.ball_positions[i]
            prev_frame_num, prev_x, prev_y, prev_conf = self.ball_positions[i - 1]

            # Calculate velocity (positive = downward, negative = upward in image coords)
            velocity = y - prev_y
            velocity_history.append(velocity)

            # Need at least 3 velocity measurements
            if len(velocity_history) < 3:
                continue

            # Check for velocity reversal (ball bouncing up after going down)
            recent_velocities = list(velocity_history)

            # Looking for pattern: positive velocities followed by negative
            if (
                recent_velocities[-3] > self.velocity_threshold
                and recent_velocities[-2] > 0
                and recent_velocities[-1] < -self.velocity_threshold
            ):
                # Check minimum time between juggles
                if frame_num - last_juggle_frame >= self.min_frames_between_juggles:
                    # Record the previous frame instead of the current frame
                    self.juggle_frames.append(prev_frame_num)
                    last_juggle_frame = frame_num

    def process_video(self, visualize=True, output_path=None):
        """
        Process entire video and detect juggles

        Args:
            visualize: Whether to display the video with annotations
            output_path: Optional path to save annotated video
        """
        frame_num = 0

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (frame_width, frame_height)
            )

        print(f"Processing video at {self.fps} fps...")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect ball
            detection = self.detect_ball(frame)

            if detection:
                x, y, conf = detection
                self.ball_positions.append((frame_num, x, y, conf))

                # Draw detection
                if visualize or writer:
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Ball: {conf:.2f}",
                        (int(x) + 15, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Show frame number
            if visualize or writer:
                cv2.putText(
                    frame,
                    f"Frame: {frame_num}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            if visualize:
                cv2.imshow("Ball Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(frame)

            frame_num += 1

            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames...")

        self.cap.release()
        if writer:
            writer.release()
        if visualize:
            cv2.destroyAllWindows()

        print(f"\nDetected ball in {len(self.ball_positions)} frames")

        # Now detect juggles from ball positions
        print("Analyzing ball trajectory for juggles...")
        self.detect_juggles()

        print(f"Detected {len(self.juggle_frames)} juggles")

    def visualize_with_juggles(self, output_path):
        """
        Create a video with juggle events marked
        """
        self.cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, (frame_width, frame_height)
        )

        frame_num = 0
        juggle_frames_set = set(self.juggle_frames)
        ball_positions_dict = {pos[0]: pos for pos in self.ball_positions}

        # For visual feedback, show juggle indicator for a few frames
        juggle_display_frames = set()
        for jf in self.juggle_frames:
            for offset in range(-2, 3):  # Show 5 frames total
                juggle_display_frames.add(jf + offset)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw ball position if available
            if frame_num in ball_positions_dict:
                _, x, y, conf = ball_positions_dict[frame_num]
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)

            # Highlight juggle events
            if frame_num in juggle_frames_set:
                cv2.putText(
                    frame,
                    "JUGGLE!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3,
                )
                # Draw a larger circle at juggle moment
                if frame_num in ball_positions_dict:
                    _, x, y, _ = ball_positions_dict[frame_num]
                    cv2.circle(frame, (int(x), int(y)), 30, (0, 0, 255), 3)

            # Show juggle indicator for nearby frames
            elif frame_num in juggle_display_frames:
                cv2.circle(frame, (50, 100), 20, (0, 165, 255), -1)

            # Show juggle count
            juggle_count = sum(1 for jf in self.juggle_frames if jf <= frame_num)
            cv2.putText(
                frame,
                f"Juggles: {juggle_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )

            writer.write(frame)
            frame_num += 1

        self.cap.release()
        writer.release()
        print(f"Saved annotated video to {output_path}")

    def export_labels(self, output_json):
        """
        Export juggle timestamps to JSON
        """
        timestamps = [frame / self.fps for frame in self.juggle_frames]

        # Calculate video length
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_length = total_frames / self.fps

        labels = {
            "video_path": str(self.video_path),
            "fps": self.fps,
            "video_length": video_length,
            "total_juggles": len(self.juggle_frames),
            "juggle_frames": self.juggle_frames,
            "juggle_timestamps": timestamps,
        }

        with open(output_json, "w") as f:
            json.dump(labels, f, indent=2)

        print(f"Exported labels to {output_json}")
        return labels
