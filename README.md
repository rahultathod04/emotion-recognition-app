<div align="center">

# 🎭 Real-Time Emotion Recognition App

### From a pre-trained library to a fully custom-trained deep learning model — built from scratch.

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93-blue?style=for-the-badge)
![Keras](https://img.shields.io/badge/Keras-Custom_CNN-D00000?style=for-the-badge&logo=keras&logoColor=white)

**Built by [Rahul Tathod](https://github.com/rahultathod04) · May 2026**

</div>

---

## 🧠 What This Project Does

This app detects **7 human emotions in real-time** from a webcam or video file using computer vision and deep learning.

It **evolved through 3 versions** — each more powerful than the last:

```
V1 → Pre-trained FER library         (plug and play)
V2 → DeepFace + MTCNN                (higher accuracy + multi-face)
V3 → Custom CNN trained on FER2013   (your own AI model)
```

**7 emotions detected:**
`😄 Happy` `😢 Sad` `😠 Angry` `😲 Surprise` `😨 Fear` `🤢 Disgust` `😐 Neutral`

---

## 🚀 The Evolution — V1 → V2 → V3

### Version 1 — FER Library (Baseline)
- Used the pre-trained **FER library** built on a CNN
- Single face detection
- Quick to set up, good as a baseline
- Accuracy: ~65% on real-world video

### Version 2 — DeepFace + MTCNN (Upgraded)
- Switched to **DeepFace** — ensemble of multiple deep learning models
- Added **MTCNN** face detector — far more accurate than basic Haar Cascade
- Added **multiple face support** — detects all faces in frame simultaneously
- Added **CSV emotion logging** with timestamps
- Accuracy: ~75% on real-world video

### Version 3 — Custom Trained CNN (Final)
- Trained a **custom CNN from scratch** on the FER2013 dataset
- 28,709 training images + 7,178 test images
- 3-block deep CNN with BatchNormalization and Dropout
- Trained on **Google Colab T4 GPU** — 50 epochs
- **Best validation accuracy: 68.01%** (saved at epoch 48)
- Full control — your own model, your own AI

---

## 📊 Model Performance

| Metric | Result |
|--------|--------|
| Training Accuracy | **71.93%** |
| Validation Accuracy | **68.01%** |
| Best Epoch | **48 / 50** |
| Training Time | ~25 min (Google Colab T4 GPU) |
| Dataset | FER2013 — 35,887 images |
| Classes | 7 emotions |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9 | Core language |
| OpenCV | Video capture, face detection, drawing overlays |
| FER Library | V1 — Pre-trained emotion model |
| DeepFace | V2 — Multi-model emotion analysis |
| MTCNN | V2 — Advanced face detection |
| TensorFlow / Keras | V3 — Custom CNN training & inference |
| NumPy | Array operations |
| CSV | Emotion logging with timestamps |
| Google Colab | GPU training environment |
| Anaconda | Environment management |

---

## 🧱 CNN Architecture (V3)

```
Input: 48×48 grayscale face image
         ↓
┌─────────────────────────────┐
│  Block 1: Conv2D(64) × 2    │  → learns edges, lines
│  BatchNorm + MaxPool + Drop │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Block 2: Conv2D(128) × 2   │  → learns eyes, nose, mouth
│  BatchNorm + MaxPool + Drop │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Block 3: Conv2D(256) × 2   │  → learns full expressions
│  BatchNorm + MaxPool + Drop │
└─────────────────────────────┘
         ↓
   Dense(512) → Dense(256)
         ↓
   Output: 7 emotion scores (softmax)
```

---

## 📁 Project Structure

```
emotion-recognition-app/
│
├── 📁 src/
│   ├── emotion_app.py        # V1 — FER Library
│   ├── emotion_app_v2.py     # V2 — DeepFace + MTCNN
│   └── emotion_app_v3.py     # V3 — Custom CNN (FINAL)
│
├── 📁 model/
│   └── emotion_model.h5      # Custom trained model weights
│
├── 📁 training/
│   └── train_model.py        # Full training script
│
├── 📁 logs/
│   └── emotion_log.csv       # Emotion detections with timestamps
│
├── 📁 demo/
│   └── frames/               # Extracted video frames for analysis
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rahultathod04/emotion-recognition-app.git
cd emotion-recognition-app
```

### 2. Create conda environment
```bash
conda create -n emotion_app python=3.9 -y
conda activate emotion_app
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run your preferred version

**Version 1 — FER (Fastest)**
```bash
python src/emotion_app.py
```

**Version 2 — DeepFace + MTCNN (Most accurate)**
```bash
python src/emotion_app_v2.py
```

**Version 3 — Custom Model (Your own AI)**
```bash
python src/emotion_app_v3.py
```

> Press **Q** to quit any version.

---

## 📚 What I Learned

- How **Convolutional Neural Networks (CNNs)** process and classify images
- How **real-time video processing** works with OpenCV frame loops
- The difference between **pre-trained models vs custom trained models**
- How **data augmentation** artificially expands training datasets
- What **Transfer Learning** means and when to use it
- How to use **Google Colab GPU** for fast model training
- How **MTCNN** achieves better face detection than Haar Cascades
- How **BatchNormalization and Dropout** prevent overfitting
- Real-world debugging — camera backend conflicts, path issues, virtual cameras
- Full **Git + GitHub** workflow for professional project management

---

## 🔮 Future Improvements

- [ ] Build a **Streamlit web app** version — runs in browser
- [ ] Train on **AffectNet** (1M+ images) for higher accuracy
- [ ] Add **attention mechanisms** to the CNN
- [ ] Real-time **emotion analytics dashboard**
- [ ] Support for **video file batch processing**
- [ ] **REST API** for emotion detection as a service
- [ ] Compare results with **Azure Face API** and **AWS Rekognition**

---

## 🔬 Comparison: Custom CNN vs Large AI Models

| | V1 FER | V2 DeepFace | V3 Custom CNN | Claude / GPT-4V |
|---|---|---|---|---|
| Speed | ⚡ Real-time | 🔄 Near real-time | ⚡ Real-time | 🐢 Seconds/frame |
| Accuracy | ~65% | ~75% | ~68% | ~85%+ |
| Works on video | ✅ | ✅ | ✅ | ❌ Frame by frame |
| Training data | 35K images | Millions | 35K (yours) | Billions |
| Cost | Free | Free | Free | Paid API |
| Runs offline | ✅ | ✅ | ✅ | ❌ |

---

## 👤 Author

**Rahul Tathod**
-  [GitHub](https://github.com/rahultathod04)
-  [linkedin](https://www.linkedin.com/in/rahul-tathod-498080409?utm_source=share_via&utm_content=profile&utm_medium=member_android)

---

## 📌 Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — Kaggle
- [FER Library](https://github.com/justinshenk/fer) — Justin Shenk
- [DeepFace](https://github.com/serengil/deepface) — Sefik Ilkin Serengil
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Google Colab](https://colab.research.google.com/) — Free GPU training

---

<div align="center">

⭐ **If you found this helpful, give it a star!** ⭐

</div>
