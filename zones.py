
import cv2
import json
import numpy as np
from pathlib import Path


ZONE_PATH = Path(__file__).parent / "config" / "zones.json"


def in_zone(point, polygon):

    if point is None or len(polygon) < 3:
        return False

    contour = np.array(polygon, dtype=np.float32)

    result = cv2.pointPolygonTest(
        contour,
        (float(point[0]), float(point[1])),
        False
    )

    return result >= 0


class ZoneManager:

    def __init__(self):

        self.points = []
        self.validated = False
        self.width = 1
        self.height = 1

        self.load()

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            if not self.validated:
                self.points.append([
                    x / self.width,
                    y / self.height
                ])

        elif event == cv2.EVENT_RBUTTONDOWN:

            if not self.validated and self.points:
                self.points.pop()

    def save(self):

        if len(self.points) < 3:
            print("Minimum 3 points necessaires.")
            return

        contour = np.array(self.points, dtype=np.float32)

        if cv2.contourArea(contour) < 0.0001:
            print("Zone invalide.")
            return

        ZONE_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )

        data = {
            "name": "restricted_zone",
            "type": "forbidden",
            "coordinate_system": "normalized",
            "polygon": self.points
        }

        with open(ZONE_PATH, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        self.validated = True

        print("Zone sauvegardee :", ZONE_PATH)

    def load(self):

        if ZONE_PATH.exists():

            try:
                with open(ZONE_PATH, "r", encoding="utf-8") as file:
                    data = json.load(file)

                points = np.array(
                    data["polygon"], dtype=np.float32
                )

                if (
                    points.ndim == 2
                    and points.shape[1] == 2
                    and len(points) >= 3
                    and np.isfinite(points).all()
                    and (points >= 0).all()
                    and (points <= 1).all()
                    and cv2.contourArea(points) > 0.0001
                ):
                    self.points = points.tolist()
                    self.validated = True
                    print("Zone chargee automatiquement.")

            except (OSError, ValueError, KeyError, TypeError):
                print("Configuration de zone invalide.")

    def reset(self):

        self.points = []
        self.validated = False

        if ZONE_PATH.exists():
            ZONE_PATH.unlink()

        print("Zone reinitialisee.")

    def contains(self, point):

        if not self.validated:
            return False

        return in_zone(point, self.points)

    def draw(self, frame, intrusion=False):

        self.height, self.width = frame.shape[:2]

        if not self.points:
            return frame

        points = (
            np.array(self.points) *
            [self.width, self.height]
        ).astype(np.int32)

        color = (0, 0, 255) if intrusion else (0, 255, 0)

        if self.validated:

            overlay = frame.copy()

            cv2.fillPoly(
                overlay, [points], color
            )

            cv2.addWeighted(
                overlay, 0.15,
                frame, 0.85, 0, frame
            )

        if len(points) >= 2:

            cv2.polylines(
                frame,
                [points],
                self.validated,
                color,
                2
            )

        for point in points:

            cv2.circle(
                frame,
                tuple(point),
                5,
                color,
                -1
            )

        return frame
