
import cv2
import time
import numpy as np
import argparse

from pose import PoseDetector


def main():

    print("Demarrage de Human Flow Analysis...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    source = args.video if args.video else 0

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la source.")
        return

    print("Source video ouverte.")

    detector = None

    try:

        detector = PoseDetector()

        print("MediaPipe initialise.")
        print("Appuyez sur Q pour quitter.")

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
                print("Fin de la video ou erreur de lecture.")
                break

            h, w = frame.shape[:2]

            # Horodatage
            if args.video:
                timestamp = int(
                    frame_index * 1000 / source_fps
                )
            else:
                timestamp = int(
                    (time.perf_counter() - start_time) * 1000
                )

            timestamp = max(timestamp, last_timestamp + 1)

            last_timestamp = timestamp
            frame_index += 1

            # Detection automatique des articulations
            keypoints = detector.extract_keypoints(
                frame, timestamp
            )

            # Fond noir pour respecter l'anonymat
            display = np.zeros_like(frame)

            if keypoints is not None:

                detector.draw_skeleton(display, keypoints)

                # Centre des hanches
                hip = detector.get_hip_center(keypoints)

                cx = int(hip[0] * w)
                cy = int(hip[1] * h)

                cv2.circle(
                    display,
                    (cx, cy),
                    8,
                    (0, 0, 255),
                    -1
                )

                status = "Pose detectee"

                cv2.putText(
                    display,
                    "33 points articulaires",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                status = "Aucune pose detectee"

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
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                status,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Affichage du resultat
            cv2.imshow(
                "Human Flow Analysis - WP1",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()

        print("Programme termine.")


if __name__ == "__main__":
    main()
