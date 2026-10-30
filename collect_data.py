
import cv2
import csv
import os
import time
import mediapipe as mp

os.makedirs("data", exist_ok=True)
file = "data/sign_data.csv"
labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    stop = False

    for label in labels:
        print(f"Show ASL sign {label}, then press S once.")
        count = 0
        collecting = False

        while count < 30:
            ret, frame = cap.read()
            if not ret:
                stop = True
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )
            result = detector.detect_for_video(
                image, int(time.monotonic() * 1000)
            )

            hand = result.hand_landmarks[0] if result.hand_landmarks else None

            if hand:
                for p in hand:
                    cv2.circle(
                        frame,
                        (int(p.x * frame.shape[1]),
                         int(p.y * frame.shape[0])),
                        4, (0, 255, 0), -1
                    )

            message = f"{label}: {count}/30 | S=Start, Q=Quit"
            cv2.putText(frame, message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            cv2.imshow("Collect A-Z Signs", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                stop = True
                break

            if key == ord("s") and hand is not None:
                collecting = True

            if collecting and hand is not None:
                row = [label]
                for p in hand:
                    row.extend([p.x, p.y, p.z])

                with open(file, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                count += 1
                time.sleep(0.08)

            if stop:
                break

        if stop:
            break

cap.release()
cv2.destroyAllWindows()
print("Collection finished.")
