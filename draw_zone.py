"""draw_zone.py : dessiner des zones à la souris et les sauver dans zones.json.

Usage :
    python draw_zone.py data/raw/video_01.mp4            # utilise la 1re image
    python draw_zone.py data/raw/video_01.mp4 --frame 50

Commandes dans la fenêtre :
    clic gauche : ajouter un sommet
    u           : annuler le dernier sommet
    Entrée      : valider le polygone (nom et type demandés dans le terminal)
    d           : supprimer la dernière zone validée
    s           : sauvegarder zones.json et quitter
    q / Échap   : quitter sans sauvegarder

Format de zones.json (coordonnées NORMALISÉES 0-1, comme les points MediaPipe) :
    [{"nom": "zone_1", "type": "interdit", "points": [[x, y], ...]}, ...]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COULEURS = {"interdit": (0, 0, 255), "autorisé": (0, 200, 0)}


def lire_image(video: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la vidéo : {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        sys.exit("Impossible de lire cette image de la vidéo.")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("zones.json"))
    args = ap.parse_args()

    base = lire_image(args.video, args.frame)
    h, w = base.shape[:2]

    zones = []
    if args.out.exists():
        try:
            zones = json.loads(args.out.read_text(encoding="utf-8"))
            print(f"{len(zones)} zone(s) existante(s) chargée(s) depuis {args.out}")
        except json.JSONDecodeError:
            zones = []
    courant = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            courant.append((x, y))

    cv2.namedWindow("draw_zone")
    cv2.setMouseCallback("draw_zone", on_mouse)

    while True:
        img = base.copy()
        for z in zones:
            pts = np.array([[int(px * w), int(py * h)] for px, py in z["points"]], np.int32)
            col = COULEURS.get(z["type"], (255, 255, 0))
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], col)
            img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
            cv2.polylines(img, [pts], True, col, 2)
            cv2.putText(img, z["nom"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if courant:
            pts = np.array(courant, np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)
            for p in courant:
                cv2.circle(img, p, 4, (0, 255, 255), -1)
        cv2.putText(img, "clic: sommet | Entree: valider | u: annuler | d: suppr zone | s: sauver | q: quitter",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("draw_zone", img)
        k = cv2.waitKey(30) & 0xFF

        if k in (ord("q"), 27):
            print("Quitté sans sauvegarder.")
            break
        elif k == ord("u") and courant:
            courant.pop()
        elif k == ord("d") and zones:
            print(f"Zone supprimée : {zones.pop()['nom']}")
        elif k == 13:  # Entrée
            if len(courant) < 3:
                print("Il faut au moins 3 sommets.")
                continue
            nom = input("Nom de la zone : ").strip() or f"zone_{len(zones) + 1}"
            typ = ""
            while typ not in ("autorisé", "interdit"):
                typ = input("Type (a = autorisé, i = interdit) : ").strip().lower()
                typ = {"a": "autorisé", "i": "interdit", "autorise": "autorisé"}.get(typ, typ)
            zones.append({
                "nom": nom,
                "type": typ,
                "points": [[round(x / w, 5), round(y / h, 5)] for x, y in courant],
            })
            courant.clear()
            print(f"Zone '{nom}' ({typ}) ajoutée.")
        elif k == ord("s"):
            args.out.write_text(json.dumps(zones, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{len(zones)} zone(s) sauvegardée(s) dans {args.out}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()