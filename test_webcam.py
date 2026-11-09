
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json

MODEL_PATH = "models/trained/isl_test_model.h5"
LABEL_PATH = "models/trained/isl_test_labels.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH) as f:
    label_map = json.load(f)

labels = {int(v): k for k, v in label_map.items()}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        prediction_text = "Show your hand"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for point in hand.landmark:
                landmarks.extend([point.x, point.y, point.z])

            X = np.array([landmarks], dtype=np.float32)

            predictions = model.predict(X, verbose=0)[0]

            index = np.argmax(predictions)
            confidence = predictions[index] * 100

            prediction_text = (
                f"{labels[index]} - {confidence:.1f}%"
            )

        cv2.putText(
            frame, prediction_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.imshow("ISL Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

