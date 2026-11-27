import os, urllib.request
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


def _get_landmarker():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_trajectory(video_path, max_gap=10):
    """Renvoie un DataFrame (t, x, y) en pixels : centre des hanches.
    Les trous de moins de max_gap images sont interpolés."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarker = _get_landmarker()
    rows, n, last_ts = [], 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        ts = max(int(t * 1000), last_ts + 1)      # millisecondes, strictement croissant
        last_ts = ts

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, ts)

        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            x = (lm[23].x + lm[24].x) / 2 * largeur   # normalisé -> pixels
            y = (lm[23].y + lm[24].y) / 2 * hauteur
        else:
            x = y = float("nan")
        rows.append((t, x, y))
        n += 1

    cap.release()
    landmarker.close()

    df = pd.DataFrame(rows, columns=["t", "x", "y"])
    print(f"{df['x'].isna().sum()} images sans détection sur {len(df)}")
    df["x"] = df["x"].interpolate(limit=max_gap, limit_direction="both")
    df["y"] = df["y"].interpolate(limit=max_gap, limit_direction="both")
    df.attrs.update(fps=fps, largeur=largeur, hauteur=hauteur)
    return df
