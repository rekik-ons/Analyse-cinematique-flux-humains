"""common.py : fonctions partagées (contrat d'interface)."""
import cv2
import numpy as np
import mediapipe as mp

_pose = None


def extract_keypoints(frame_bgr) -> np.ndarray | None:
    """Image OpenCV (BGR) -> squelette (33, 3) = x, y, visibilité, ou None si personne."""
    global _pose
    if _pose is None:
        _pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    res = _pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None
    return np.array([[l.x, l.y, l.visibility] for l in res.pose_landmarks.landmark],
                    dtype=np.float32)


def hip_center(kp: np.ndarray) -> tuple[float, float]:
    """Moyenne des points 23 et 24 -> (x, y)."""
    return float(kp[[23, 24], 0].mean()), float(kp[[23, 24], 1].mean())


def normalize(kp: np.ndarray) -> np.ndarray:
    """(33, 3) -> (33, 2) : centré sur les hanches, divisé par la taille du tronc."""
    hx, hy = hip_center(kp)
    sx, sy = float(kp[[11, 12], 0].mean()), float(kp[[11, 12], 1].mean())
    tronc = np.hypot(sx - hx, sy - hy)
    if not np.isfinite(tronc) or tronc < 1e-3:
        return np.full((33, 2), np.nan, dtype=np.float32)
    return ((kp[:, :2] - [hx, hy]) / tronc).astype(np.float32)