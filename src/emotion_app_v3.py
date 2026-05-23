"""
Real-Time Emotion Recognition App - Version 3.0 (FINAL)
---------------------------------------------------------
Uses YOUR OWN custom trained CNN model (emotion_model.h5)
- 7 emotions detected
- Multiple faces simultaneously  
- Live emotion bar chart
- CSV emotion log with timestamps

Press Q to quit.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import csv
from datetime import datetime

# ── 1. Load YOUR custom trained model ────────────────────────────────────────

print("⏳ Loading your custom model...")
model = load_model("emotion_model.h5")
print("✅ Model loaded!")

# Emotion labels — must match training order
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Colors per emotion (BGR)
EMOTION_COLORS = {
    "happy":    (50,  220, 100),
    "sad":      (200,  80,  50),
    "angry":    (50,   50, 230),
    "surprise": (50,  200, 255),
    "fear":     (180,  50, 180),
    "disgust":  (50,  160,  50),
    "neutral":  (180, 180, 180),
}

EMOTION_EMOJI = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "surprise": "😲", "fear": "😨", "disgust": "🤢", "neutral": "😐"
}

# ── 2. Load face detector ─────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── 3. Open video source ──────────────────────────────────────────────────────

cap = cv2.VideoCapture("rajpalji.mp4")   # change to 0 for webcam

if not cap.isOpened():
    print("❌ Could not open video source.")
    exit()

print("✅ Video started! Press Q to quit.\n")

# ── 4. Setup CSV log ──────────────────────────────────────────────────────────

csv_file   = open("emotion_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Face_ID", "Emotion", "Confidence"])
print("📝 Logging to emotion_log.csv\n")

# ── 5. Helper: draw emotion bars ──────────────────────────────────────────────

def draw_emotion_bars(frame, predictions, start_x, start_y):
    bar_max = 120
    bar_h   = 15
    gap     = 22

    for i, (emotion, score) in enumerate(zip(EMOTIONS, predictions)):
        y     = start_y + i * gap
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        bar_w = int(score * bar_max)

        # Background
        cv2.rectangle(frame, (start_x, y),
                      (start_x + bar_max, y + bar_h), (50, 50, 50), -1)
        # Filled bar
        if bar_w > 0:
            cv2.rectangle(frame, (start_x, y),
                          (start_x + bar_w, y + bar_h), color, -1)
        # Label
        cv2.putText(frame, f"{emotion[:3].upper()} {score*100:.0f}%",
                    (start_x + bar_max + 6, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


# ── 6. Main loop ──────────────────────────────────────────────────────────────

prev_emotions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video finished.")
        break

    frame     = cv2.flip(frame, 1)
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Detect all faces in frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors= 5,
        minSize     = (30, 30)
    )

    for face_id, (x, y, w, h) in enumerate(faces):

        # Crop and preprocess face for model
        roi        = gray[y:y+h, x:x+w]
        roi        = cv2.resize(roi, (48, 48))
        roi        = roi.astype("float32") / 255.0
        roi        = img_to_array(roi)
        roi        = np.expand_dims(roi, axis=0)

        # Predict emotion with YOUR model
        predictions = model.predict(roi, verbose=0)[0]
        top_idx     = np.argmax(predictions)
        top_emotion = EMOTIONS[top_idx]
        top_score   = predictions[top_idx]
        color       = EMOTION_COLORS.get(top_emotion, (255, 255, 255))

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Face ID
        cv2.putText(frame, f"Face {face_id+1}",
                    (x, y-30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        # Emotion label
        label      = f"{top_emotion.upper()}  {top_score*100:.0f}%"
        lsize, _   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x, y-lsize[1]-16), (x+lsize[0]+12, y), color, -1)
        cv2.putText(frame, label, (x+6, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)

        # Emotion bars
        bars_x = x + w + 10
        if bars_x + 200 < frame.shape[1]:
            draw_emotion_bars(frame, predictions, bars_x, y)

        # Print + log only when emotion changes
        if prev_emotions.get(face_id) != top_emotion:
            emoji = EMOTION_EMOJI.get(top_emotion, "")
            print(f"  {emoji} Face {face_id+1}: {top_emotion.upper()} ({top_score*100:.0f}%)")
            csv_writer.writerow([timestamp, face_id+1, top_emotion, f"{top_score*100:.1f}"])
            prev_emotions[face_id] = top_emotion

    # Watermark
    cv2.putText(frame, "Emotion AI v3.0 | Custom Model by Rahul Tathod",
                (10, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Emotion Recognition v3.0  |  Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n👋 App closed.")
        break

# ── 7. Cleanup ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("✅ Emotion log saved to emotion_log.csv")
