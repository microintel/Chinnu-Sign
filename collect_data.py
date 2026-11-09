
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

DATA_DIR = "data/isl_landmarks"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "isl_alphabet.csv")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam. Check your camera.")
    exit()

# Create CSV with 63 landmark features and one label
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [f"f{i}" for i in range(63)] + ["label"]
        )

print("ISL A-Z Data Collection")
print("Press A-Z to select a letter.")
print("Press SPACE to save a sample.")
print("Press Q to quit.")

current_letter = "A"
samples_saved = {letter: 0 for letter in letters}

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for point in hand.landmark:
                landmarks.extend([point.x, point.y, point.z])

        cv2.putText(
            frame, f"Letter: {current_letter}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Saved: {samples_saved[current_letter]}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

        cv2.putText(
            frame, "SPACE=Save | A-Z=Select | Q=Quit",
            (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )

        cv2.imshow("ISL Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == 32 and landmarks is not None:
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmarks + [current_letter])

            samples_saved[current_letter] += 1
            print(f"Saved {current_letter}: {samples_saved[current_letter]}")
            time.sleep(0.2)

        elif key != 255:
            character = chr(key).upper()
            if character in letters:
                current_letter = character

cap.release()
cv2.destroyAllWindows()

print("Collection finished.")
print("Dataset saved at:", CSV_FILE)

