"""extract_points.py : vidéo -> CSV des 33 points MediaPipe (contrat d'interface).

Usage :
    python extract_points.py data/raw/video_01.mp4
    python extract_points.py data/raw/video_01.mp4 --show        # aperçu à l'écran
    python extract_points.py data/raw/video_01.mp4 --out data/points

Sortie : data/points/video_01_points.csv
Colonnes : frame, t, person_id, x0, y0, v0, ..., x32, y32, v32
Coordonnées normalisées 0-1, v = visibilité. Image sans personne : x, y, v = NaN.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

NB_POINTS = 33


def header():
    cols = ["frame", "t", "person_id"]
    for i in range(NB_POINTS):
        cols += [f"x{i}", f"y{i}", f"v{i}"]
    return cols


def extract(video_path: Path, out_dir: Path, show: bool = False, complexity: int = 1) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{video_path.stem}_points.csv"

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    nan_row = ["nan"] * (NB_POINTS * 3)

    detected = 0
    frame_idx = 0
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header())

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            t = round(frame_idx / fps, 4)

            if res.pose_landmarks:
                detected += 1
                row = []
                for lm in res.pose_landmarks.landmark:
                    row += [round(lm.x, 5), round(lm.y, 5), round(lm.visibility, 3)]
            else:
                row = nan_row
            writer.writerow([frame_idx, t, 0] + row)

            if show:
                if res.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("extract_points (q pour quitter)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{total} images", end="\r")

    cap.release()
    cv2.destroyAllWindows()
    taux = 100 * detected / max(frame_idx, 1)
    print(f"\nOK : {frame_idx} images, personne détectée sur {taux:.1f} % -> {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Extraction des 33 points MediaPipe d'une vidéo")
    ap.add_argument("video", type=Path, help="chemin de la vidéo (ex: data/raw/video_01.mp4)")
    ap.add_argument("--out", type=Path, default=Path("data/points"), help="dossier de sortie")
    ap.add_argument("--show", action="store_true", help="afficher le squelette pendant l'extraction")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="0 = rapide, 2 = précis")
    args = ap.parse_args()
    extract(args.video, args.out, args.show, args.complexity)


if __name__ == "__main__":
    main()