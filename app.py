
import cv2
import time
import joblib
import mediapipe as mp
from collections import Counter, deque

# Load trained ML model
model = joblib.load("sign_model.pkl")

# Store recent predictions to reduce flickering
recent_predictions = deque(maxlen=7)

# Configure hand detector
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not available!")
    raise SystemExit

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Unable to read camera!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect_for_video(
            image, int(time.monotonic() * 1000)
        )

        text = "Show your hand"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            features = []

            # Extract hand landmark coordinates
            for point in hand:
                features.extend([point.x, point.y, point.z])

                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])

                cv2.circle(
                    frame, (x, y), 4, (0, 255, 0), -1
                )

            # Predict letter
            letter = model.predict([features])[0]
            recent_predictions.append(letter)

            # Display a stable prediction after 7 frames
            if len(recent_predictions) == 7:
                stable_letter = Counter(
                    recent_predictions
                ).most_common(1)[0][0]

                text = "Predicted Letter: " + str(stable_letter)
            else:
                text = "Recognizing..."

        else:
            recent_predictions.clear()

        # Display result
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 65),
                      (30, 30, 30), -1)

        cv2.putText(
            frame, text, (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            (0, 255, 0), 2
        )

        cv2.imshow("Real-Time Sign Language Recognition", frame)

        # Press Q to close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Application closed.")
