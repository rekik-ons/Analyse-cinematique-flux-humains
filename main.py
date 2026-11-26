
import cv2
import time
import numpy as np
import argparse

from pose import PoseDetector
from zones import ZoneManager


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--video", default=None)

    args = parser.parse_args()

    source = args.video if args.video else 0

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        print("Erreur : source video inaccessible.")

        return

    detector = None

    try:

        detector = PoseDetector()
        zone = ZoneManager()

        window_name = "Human Flow Analysis - WP2"

        cv2.namedWindow(window_name)

        cv2.setMouseCallback(
            window_name,
            zone.mouse_callback
        )

        print("Clic gauche : ajouter un sommet")
        print("Clic droit : supprimer le dernier sommet")
        print("S : sauvegarder la zone")
        print("R : reinitialiser la zone")
        print("Q : quitter")

        start_time = time.perf_counter()
        previous_time = start_time

        last_timestamp = -1
        frame_index = 0

        source_fps = cap.get(cv2.CAP_PROP_FPS)

        if source_fps <= 0:
            source_fps = 30

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            h, w = frame.shape[:2]

            if args.video:

                timestamp = int(
                    frame_index * 1000 / source_fps
                )

            else:

                timestamp = int(
                    (time.perf_counter() - start_time) * 1000
                )

            timestamp = max(
                timestamp,
                last_timestamp + 1
            )

            last_timestamp = timestamp
            frame_index += 1

            keypoints = detector.extract_keypoints(
                frame, timestamp
            )

            # Affichage anonymise
            display = np.zeros_like(frame)

            hip = None
            intrusion = False

            status = "Aucune pose detectee"
            color = (255, 255, 255)

            if keypoints is not None:

                hip = detector.get_hip_center(
                    keypoints
                )

                intrusion = zone.contains(hip)

                if not zone.validated:

                    status = "Zone non configuree"
                    color = (0, 255, 255)

                elif intrusion:

                    status = "INTRUSION"
                    color = (0, 0, 255)

                else:

                    status = "AUTORISE"
                    color = (0, 255, 0)

            # Dessiner la zone
            zone.draw(display, intrusion)

            # Dessiner le squelette
            if keypoints is not None:

                detector.draw_skeleton(
                    display, keypoints
                )

                cx = int(hip[0] * w)
                cy = int(hip[1] * h)

                cv2.circle(
                    display,
                    (cx, cy),
                    8,
                    color,
                    -1
                )

            # Calcul des FPS
            current_time = time.perf_counter()

            fps = 1 / max(
                current_time - previous_time,
                0.000001
            )

            previous_time = current_time

            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                status,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            if intrusion:

                cv2.rectangle(
                    display,
                    (2, 2),
                    (w - 3, h - 3),
                    (0, 0, 255),
                    5
                )

            cv2.putText(
                display,
                "S: Save | R: Reset | Q: Quit",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            cv2.imshow(window_name, display)

            delay = (
                max(1, int(1000 / source_fps))
                if args.video else 1
            )

            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                zone.save()

            elif key == ord("r"):
                zone.reset()

    finally:

        cap.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
