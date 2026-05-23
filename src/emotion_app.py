"""
Real-Time Emotion Recognition App
-----------------------------------
Uses your webcam to detect faces and predict emotions live.
Press Q to quit the app.
"""

import cv2
from fer import FER

# ── 1. Setup ──────────────────────────────────────────────────────────────────

# Load the emotion detector (set mtcnn=True for better accuracy, but slower)
detector = FER(mtcnn=True)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture("rajpalji.mp4")

if not cap.isOpened():
    print("❌ Could not open webcam. Check your camera connection.")
    exit()

print("⏳ Warming up camera...")
for _ in range(30):
    cap.read()

print("✅ Webcam started! Press Q to quit.")


# Color for each emotion (in BGR format for OpenCV)
EMOTION_COLORS = {
    "happy":    (50,  220, 100),   # green
    "sad":      (200,  80,  50),   # blue-ish
    "angry":    (50,   50, 230),   # red
    "surprise": (50,  200, 255),   # yellow
    "fear":     (180,  50, 180),   # purple
    "disgust":  (50,  160,  50),   # dark green
    "neutral":  (180, 180, 180),   # gray
}

# Emoji for each emotion (shown in terminal too)
EMOTION_EMOJI = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "surprise": "😲", "fear": "😨", "disgust": "🤢", "neutral": "😐"
}


# ── 2. Helper: draw the emotion bar chart on screen ───────────────────────────

def draw_emotion_bars(frame, emotions, start_x, start_y):
    """Draws a small bar chart of all emotion scores beside the face."""
    bar_max_width = 120
    bar_height    = 16
    gap           = 22

    for i, (emotion, score) in enumerate(emotions.items()):
        y = start_y + i * gap
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        bar_width = int(score * bar_max_width)

        # Background bar (dark)
        cv2.rectangle(frame,
                      (start_x, y),
                      (start_x + bar_max_width, y + bar_height),
                      (50, 50, 50), -1)

        # Filled bar (colored by emotion)
        if bar_width > 0:
            cv2.rectangle(frame,
                          (start_x, y),
                          (start_x + bar_width, y + bar_height),
                          color, -1)

        # Label text
        label = f"{emotion[:3].upper()}  {score:.0%}"
        cv2.putText(frame, label,
                    (start_x + bar_max_width + 6, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)


# ── 3. Main loop ──────────────────────────────────────────────────────────────

prev_emotion = None  # Track changes for terminal output

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Mirror the frame so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Detect emotions in the frame
    results = detector.detect_emotions(frame)

    # ── 4. Draw results for each detected face ────────────────────────────────
    for face in results:
        x, y, w, h = face["box"]
        emotions    = face["emotions"]

        # Find the top emotion
        top_emotion = max(emotions, key=emotions.get)
        top_score   = emotions[top_emotion]
        color       = EMOTION_COLORS.get(top_emotion, (255, 255, 255))

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw emotion label above the box
        label = f"{top_emotion.upper()}  {top_score:.0%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        lw, lh = label_size

        # Label background pill
        cv2.rectangle(frame,
                      (x, y - lh - 16),
                      (x + lw + 12, y),
                      color, -1)
        cv2.putText(frame, label,
                    (x + 6, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)

        # Draw emotion bars to the right of the face box
        bars_x = x + w + 10
        if bars_x + 200 < frame.shape[1]:   # only if there's room
            draw_emotion_bars(frame, emotions, bars_x, y)

        # Print to terminal when emotion changes
        if top_emotion != prev_emotion:
            emoji = EMOTION_EMOJI.get(top_emotion, "")
            print(f"  Detected: {emoji}  {top_emotion.upper()} ({top_score:.0%})")
            prev_emotion = top_emotion

    # ── 5. Instructions overlay ───────────────────────────────────────────────
    cv2.putText(frame, "Press Q to quit",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Show the frame
    cv2.imshow("Real-Time Emotion Recognition  |  Press Q to quit", frame)

    # Quit on Q key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n👋 App closed.")
        break


# ── 6. Cleanup ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()