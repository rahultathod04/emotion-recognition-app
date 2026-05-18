# 😄 Real-Time Emotion Recognition App

A computer vision project that detects human emotions in real-time using a webcam or video file. Built with OpenCV and Deep Learning.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![FER](https://img.shields.io/badge/FER-22.5-red)

---

## 🎯 What It Does

- 📷 Opens a webcam or video file in real time
- 🧠 Detects faces automatically using deep learning
- 😄 Predicts **7 emotions** for each detected face:
  - Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- 📊 Displays a live emotion bar chart beside the face
- 🏷️ Shows the top emotion + confidence % as an overlay

---

## 🖥️ Demo

> App detects face → draws bounding box → shows emotion label + live bar chart

```
Detected: 😄 HAPPY (87%)
Detected: 😠 ANGRY (74%)
Detected: 😲 SURPRISE (91%)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9 | Core language |
| OpenCV | Webcam access & drawing overlays |
| FER Library | Emotion detection model |
| TensorFlow | Deep learning backend for FER |
| NumPy | Numerical operations |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rahultathod04/emotion-recognition-app.git
cd emotion-recognition-app
```

### 2. Create a conda environment
```bash
conda create -n emotion_app python=3.9 -y
conda activate emotion_app
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app

**With webcam:**
```bash
python emotion_app.py
```

**With a video file:**
Change this line in `emotion_app.py`:
```python
cap = cv2.VideoCapture("myvideo.mp4")
```

Press **Q** to quit.

---

## 📁 Project Structure

```
emotion-recognition-app/
│
├── emotion_app.py       # Main application
├── requirements.txt     # All dependencies
└── README.md            # Project documentation
```

---

## 📚 What I Learned

- How to capture and process real-time video frames using **OpenCV**
- How **Facial Emotion Recognition (FER)** models work under the hood
- How deep learning models detect faces and classify expressions
- How to draw overlays (bounding boxes, text, bar charts) on video frames
- Setting up a proper **Python environment** with Conda
- Debugging real-world issues like camera backend conflicts on Windows

---

## 🚀 Future Improvements

- [ ] Switch to **DeepFace** for higher accuracy
- [ ] Add **MTCNN** face detection for better face tracking
- [ ] Save emotion log to a **CSV file** with timestamps
- [ ] Build a **Streamlit web app** version
- [ ] Add support for **multiple faces** simultaneously
- [ ] Train a **custom model** on a larger dataset

---

## 👤 Author

**Rahul Tathod**
- GitHub: [@rahultathod04](https://github.com/rahultathod04)

---

## 📌 Acknowledgements

- [FER Library](https://github.com/justinshenk/fer) by Justin Shenk
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
