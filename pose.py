
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

MODEL_PATH = str(
    Path(__file__).parent / "models" / "pose_landmarker_lite.task"
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32)
]


class PoseDetector:

    def __init__(self):

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = (
            mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            )
        )

    def extract_keypoints(self, frame, timestamp_ms):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect_for_video(
            image, timestamp_ms
        )

        if not result.pose_landmarks:
            return None

        return np.array(
            [[p.x, p.y] for p in result.pose_landmarks[0]],
            dtype=np.float32
        )

    def draw_skeleton(self, frame, keypoints):

        if keypoints is None:
            return frame

        h, w = frame.shape[:2]

        points = (keypoints * [w, h]).astype(int)

        for a, b in CONNECTIONS:
            cv2.line(
                frame,
                tuple(points[a]),
                tuple(points[b]),
                (255, 255, 255),
                2
            )

        for point in points:
            cv2.circle(
                frame,
                tuple(point),
                4,
                (0, 255, 0),
                -1
            )

        return frame

    def get_hip_center(self, keypoints):

        if keypoints is None:
            return None

        return (keypoints[23] + keypoints[24]) / 2

    def close(self):
        self.detector.close()
