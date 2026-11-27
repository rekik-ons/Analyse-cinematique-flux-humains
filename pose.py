import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'models/pose_landmarker_lite.task'
CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,19),(15,21),(17,19),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(27,31),(28,30),(30,32),(28,32)]

class PoseDetector:
    def __init__(self):
        options = mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)), running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.visibility = None
    def extract_keypoints(self, frame, timestamp_ms):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            self.visibility = None
            return None
        points = result.pose_landmarks[0]
        self.visibility = np.array([p.visibility for p in points])
        return np.array([[p.x,p.y] for p in points], dtype=np.float32)
    def get_hip_center(self, keypoints):
        if keypoints is None or self.visibility is None or min(self.visibility[23], self.visibility[24]) < 0.5:
            return None
        hip = (keypoints[23]+keypoints[24])/2
        return hip if np.isfinite(hip).all() and ((hip>=0)&(hip<=1)).all() else None
    def draw_skeleton(self, frame, keypoints):
        if keypoints is None: return frame
        h,w = frame.shape[:2]
        valid = (self.visibility >= .5) & np.isfinite(keypoints).all(axis=1) & ((keypoints>=0)&(keypoints<=1)).all(axis=1)
        pts = np.clip(np.nan_to_num(keypoints),0,1)*[w-1,h-1]
        pts = pts.astype(int)
        for a,b in CONNECTIONS:
            if valid[a] and valid[b]: cv2.line(frame,tuple(pts[a]),tuple(pts[b]),(255,255,255),2)
        for i,p in enumerate(pts):
            if valid[i]: cv2.circle(frame,tuple(p),3,(0,255,0),-1)
        return frame
    def close(self): self.detector.close()
