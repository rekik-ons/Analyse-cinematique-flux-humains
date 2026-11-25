"""annotate.py : vidéo annotée (squelette, zones, boîte, identifiant, état autorisé/interdit).

Usage :
    python annotate.py data/raw/video_01.mp4
    python annotate.py data/raw/video_01.mp4 --zones zones.json --out outputs

Lit : data/points/video_01_points.csv (produit par extract_points.py) et zones.json.
Écrit : outputs/video_01_annotee.mp4

Seul le squelette est dessiné ; --masque remplace l'image par un fond noir
(respect de la vie privée : aucun visage visible).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

SEUIL_VISIBILITE = 0.5
HANCHE_G, HANCHE_D = 23, 24
CONNEXIONS = [(a.value if hasattr(a, "value") else a, b.value if hasattr(b, "value") else b)
              for a, b in mp.solutions.pose.POSE_CONNECTIONS]


def charger_points(csv_path: Path):
    """Renvoie un dict frame -> tableau (33, 3) (x, y, v) ou None si personne."""
    df = pd.read_csv(csv_path)
    par_frame = {}
    for _, r in df.iterrows():
        vals = r.iloc[3:].to_numpy(dtype=float).reshape(33, 3)
        par_frame[int(r["frame"])] = None if np.isnan(vals).all() else vals
    return par_frame


def charger_zones(path: Path, w: int, h: int):
    if not path.exists():
        return []
    zones = json.loads(path.read_text(encoding="utf-8"))
    for z in zones:
        z["px"] = np.array([[x * w, y * h] for x, y in z["points"]], np.float32)
    return zones


def etat_point(point_px, zones):
    """'INTERDIT' si le point est dans une zone interdite, sinon 'AUTORISE'."""
    for z in zones:
        if z["type"].lower().startswith("interdit"):
            if cv2.pointPolygonTest(z["px"], point_px, False) >= 0:
                return "INTERDIT", z["nom"]
    return "AUTORISE", None


def centre_hanches(kp):
    """Moyenne des points 23 et 24, ou None si peu visibles."""
    g, d = kp[HANCHE_G], kp[HANCHE_D]
    ok = [p for p in (g, d) if p[2] >= SEUIL_VISIBILITE]
    if not ok:
        return None
    return float(np.mean([p[0] for p in ok])), float(np.mean([p[1] for p in ok]))


def dessiner(img, kp, zones, person_id=0):
    h, w = img.shape[:2]
    # zones
    for z in zones:
        interdit = z["type"].lower().startswith("interdit")
        col = (0, 0, 255) if interdit else (0, 200, 0)
        pts = z["px"].astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        cv2.polylines(img, [pts], True, col, 2)
        cv2.putText(img, z["nom"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    if kp is None:
        return img, None

    vis = kp[:, 2] >= SEUIL_VISIBILITE
    pts_px = (kp[:, :2] * [w, h]).astype(int)

    # squelette
    for a, b in CONNEXIONS:
        if vis[a] and vis[b]:
            cv2.line(img, tuple(pts_px[a]), tuple(pts_px[b]), (255, 200, 0), 2)
    for i in np.where(vis)[0]:
        cv2.circle(img, tuple(pts_px[i]), 3, (0, 255, 255), -1)

    # état
    c = centre_hanches(kp)
    etat, nom_zone = "INCONNU", None
    col = (200, 200, 200)
    if c is not None:
        c_px = (c[0] * w, c[1] * h)
        etat, nom_zone = etat_point(c_px, zones)
        col = (0, 0, 255) if etat == "INTERDIT" else (0, 200, 0)
        cv2.circle(img, (int(c_px[0]), int(c_px[1])), 6, col, -1)

    # boîte englobante (points visibles uniquement)
    if vis.sum() >= 4:
        x0, y0 = pts_px[vis].min(axis=0)
        x1, y1 = pts_px[vis].max(axis=0)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        label = f"ID {person_id} | {etat}"
        cv2.putText(img, label, (x0, max(y0 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return img, (etat, nom_zone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--points", type=Path, default=None, help="CSV de points (défaut : data/points/<video>_points.csv)")
    ap.add_argument("--zones", type=Path, default=Path("zones.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--masque", action="store_true", help="fond noir : seul le squelette est visible")
    args = ap.parse_args()

    csv_path = args.points or Path("data/points") / f"{args.video.stem}_points.csv"
    if not csv_path.exists():
        sys.exit(f"CSV introuvable : {csv_path}. Lancez d'abord extract_points.py.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la vidéo : {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    points = charger_points(csv_path)
    zones = charger_zones(args.zones, w, h)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.video.stem}_annotee.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.masque:
            frame = np.zeros_like(frame)
        frame, _ = dessiner(frame, points.get(idx), zones)
        cv2.putText(frame, f"t={idx / fps:.2f}s", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"OK : {idx} images -> {out_path}")


if __name__ == "__main__":
    main()