import cv2
import numpy as np
from ultralytics import YOLO
import polars as pl


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
        self.ball_positions = pl.DataFrame()

    def detect_ball(self, frame):
        """
        Detect soccer ball in a frame.
        Returns: (x, y, confidence) or None if not detected or below threshold.
        """
        results = self.model(
            frame, classes=[32], conf=self.confidence_threshold, verbose=False
        )

        if len(results[0].boxes) == 0:
            return None

        # Choose the detection with the highest confidence
        boxes = results[0].boxes
        confidences = [b.conf[0].cpu().item() for b in boxes]
        max_idx = int(np.argmax(confidences))
        box = boxes[max_idx]
        confidence = float(box.conf[0].cpu().item())

        # Enforce threshold in case model returns lower confidences
        if confidence < self.confidence_threshold:
            return None

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Calculate center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return center_x, center_y, confidence

    def process_video(self, visualize=True, output_path=None) -> None:
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

        detections = []  # accumulate per-frame dicts
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect ball
            detection = self.detect_ball(frame)

            if detection:
                x, y, conf = detection
                detections.append(
                    {
                        "Frame": frame_num,
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(conf),
                    }
                )

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
            else:
                # No detection for this frame — still record the frame with nulls
                detections.append(
                    {"Frame": frame_num, "x": None, "y": None, "confidence": None}
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

        # Convert raw detections to a Polars DataFrame and store in self.ball_positions
        self.ball_positions = pl.DataFrame(detections).with_columns(
            [
                pl.col("Frame").cast(pl.Int64),
                pl.col("x").cast(pl.Float64),
                pl.col("y").cast(pl.Float64),
                pl.col("confidence").cast(pl.Float64),
            ]
        )
        return None
