"""
Real-Time Emotion Recognition App - Version 2.0
-------------------------------------------------
Upgrades:
- DeepFace for higher accuracy
- MTCNN for better face detection
- Multiple faces simultaneously
- Emotion log saved to CSV with timestamps

Press Q to quit.
"""

import cv2
from deepface import DeepFace
import csv
import time
from datetime import datetime

# ── 1. Setup ──────────────────────────────────────────────────────────────────

# Color for each emotion (BGR format)
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

# Open webcam / video file
cap = cv2.VideoCapture("rajpalji.mp4")  # change to 0 for webcam

if not cap.isOpened():
    print("❌ Could not open video source.")
    exit()

print("✅ Started! Press Q to quit.")

# ── 2. Setup CSV log ───────────────────────────────────────────────────────────

csv_file = open("emotion_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Face_ID", "Emotion", "Confidence"])
print("📝 Logging emotions to emotion_log.csv")

# ── 3. Helper: draw emotion bars ──────────────────────────────────────────────

def draw_emotion_bars(frame, emotions, start_x, start_y):
    bar_max_width = 100
    bar_height    = 14
    gap           = 20

    for i, (emotion, score) in enumerate(emotions.items()):
        y     = start_y + i * gap
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        bar_w = int(score / 100 * bar_max_width)

        cv2.rectangle(frame, (start_x, y), (start_x + bar_max_width, y + bar_height), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (start_x, y), (start_x + bar_w, y + bar_height), color, -1)

        label = f"{emotion[:3].upper()} {score:.0f}%"
        cv2.putText(frame, label, (start_x + bar_max_width + 6, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


# ── 4. Main loop ──────────────────────────────────────────────────────────────

frame_count  = 0
analyze_every = 5   # analyze every 5th frame for speed
last_results  = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video finished.")
        break

    frame       = cv2.flip(frame, 1)
    frame_count += 1

    # Only run DeepFace every N frames (for speed)
    if frame_count % analyze_every == 0:
        try:
            results = DeepFace.analyze(
                frame,
                actions      = ["emotion"],
                enforce_detection = False,
                detector_backend  = "mtcnn",   # MTCNN for better face detection
                silent       = True
            )
            last_results = results if isinstance(results, list) else [results]
        except Exception:
            last_results = []

    # ── 5. Draw results ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%H:%M:%S")

    for face_id, face in enumerate(last_results):
        region   = face.get("region", {})
        x        = region.get("x", 0)
        y        = region.get("y", 0)
        w        = region.get("w", 0)
        h        = region.get("h", 0)
        emotions = face.get("emotion", {})

        if not emotions:
            continue

        top_emotion = max(emotions, key=emotions.get)
        top_score   = emotions[top_emotion]
        color       = EMOTION_COLORS.get(top_emotion, (255, 255, 255))

        # Face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Face ID label
        id_label = f"Face {face_id + 1}"
        cv2.putText(frame, id_label, (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Emotion label
        label = f"{top_emotion.upper()}  {top_score:.0f}%"
        lsize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x, y - lsize[1] - 16), (x + lsize[0] + 12, y), color, -1)
        cv2.putText(frame, label, (x + 6, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)

        # Emotion bars
        bars_x = x + w + 10
        if bars_x + 180 < frame.shape[1]:
            draw_emotion_bars(frame, emotions, bars_x, y)

        # Log to CSV
        if frame_count % analyze_every == 0:
            emoji = EMOTION_EMOJI.get(top_emotion, "")
            print(f"  {emoji} Face {face_id+1}: {top_emotion.upper()} ({top_score:.0f}%)")
            csv_writer.writerow([timestamp, face_id + 1, top_emotion, f"{top_score:.1f}"])

    # Instructions
    cv2.putText(frame, "Press Q to quit",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Emotion Recognition v2.0  |  DeepFace + MTCNN", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n👋 App closed.")
        break

# ── 6. Cleanup ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("✅ Emotion log saved to emotion_log.csv")
