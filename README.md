# VisionGuard — Real-Time Deep Learning Surveillance System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)

A production-grade real-time facial recognition system for surveillance applications — malls, airports, and residential security. Built with a custom CNN trained using **Triplet Loss** and **FaceNet 128-D embeddings**, with **One-Shot Learning** for instant registration of new individuals.

---

## Demo

| Recognised ✅ | Unknown 🚨 |
|---|---|
| Green bounding box + name | Red bounding box + alert triggered |

---

## Architecture

```
IP Camera / Webcam
       │
       ▼
  OpenCV Frame Capture
       │
       ▼
  Face Detection (OpenCV DNN ResNet SSD)
       │
       ▼
  CNN Feature Extraction (TensorFlow + Keras)
       │
       ▼
  FaceNet → 128-D L2-Normalised Embedding
       │
       ▼
  Cosine Distance Matching
  (Triplet Loss trained threshold = 0.55)
       │
      / \
     /   \
  KNOWN  UNKNOWN
  Access  Alert +
  Granted Store Face
```

### Key technical choices

| Component | Choice | Why |
|---|---|---|
| Face detector | OpenCV DNN (ResNet SSD) | Fast, accurate, handles partial occlusion |
| Feature extractor | Custom CNN (4 conv blocks, 256 filters) | Real-time inference on CPU |
| Embedding | FaceNet 128-D | Compact, proven on face verification tasks |
| Training loss | Triplet Loss | Forces intra-class clustering, inter-class separation |
| Few-shot support | One-Shot Learning | Single image sufficient to register a new person |
| Distance metric | Cosine similarity | Invariant to magnitude; works perfectly with L2-normalised embeddings |

---

## Features

- **Real-time recognition** from IP cameras or webcams at live video framerates
- **97% accuracy** across varied lighting, masks, and glasses
- **One-shot registration** — register a new person with a single photo via the web UI
- **Alert system** — unknown faces trigger alerts and are saved for review
- **Live dashboard** — MJPEG stream with bounding boxes, recognition logs, and statistics
- **SQLite storage** — no external database required
- **Email alerts** — optional SMTP notification on unknown face detection

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/pranavsahtih2104/VisionGuard.git
cd VisionGuard
pip install -r requirements.txt
```

### 2. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 3. Register a person

- Go to the dashboard
- Enter a name and upload a face photo
- Click **Register** — the system will extract the 128-D embedding and store it

### 4. Use an IP camera (optional)

```bash
export IP_CAMERA_URL="rtsp://username:password@192.168.1.100:554/stream"
python app.py
```

---

## Training your own model

If you want to fine-tune the CNN on your own face dataset:

### Dataset structure

```
data/faces/
    PersonA/
        img1.jpg
        img2.jpg
        img3.jpg
    PersonB/
        img1.jpg
        img2.jpg
```

### Train

```bash
python train/train_triplet.py --data_dir data/faces --epochs 30
```

The script:
1. Samples (anchor, positive, negative) triplets from your dataset
2. Trains the CNN with Triplet Loss: `L = max(0, d(a,p) - d(a,n) + margin)`
3. Saves best weights to `models/facenet_weights.h5`

### Triplet Loss explained

```
Anchor   ─── same person ───► Positive    (should be CLOSE)
Anchor   ─── diff person ───► Negative    (should be FAR)

Loss = max(0,  d(anchor, positive)  −  d(anchor, negative)  +  margin)
```

The network learns to pull same-identity embeddings together and push different-identity embeddings apart, making recognition robust even under appearance variation.

---

## Project structure

```
VisionGuard/
├── app.py                    # Flask server + routes
├── requirements.txt
├── src/
│   ├── recognizer.py         # Core detection → embedding → matching pipeline
│   └── alert.py              # Alert system (DB + email)
├── utils/
│   ├── embedder.py           # FaceNet CNN (TensorFlow/Keras)
│   └── detector.py           # OpenCV DNN / Haar cascade detector
├── train/
│   └── train_triplet.py      # Triplet loss training script
├── database/
│   └── db.py                 # SQLite ORM
├── templates/
│   └── index.html            # Live dashboard UI
└── models/                   # Weights stored here (gitignored)
```

---

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `IP_CAMERA_URL` | RTSP URL of IP camera | Uses webcam (index 0) |
| `ALERT_SMTP_HOST` | SMTP server for email alerts | Disabled |
| `ALERT_SMTP_USER` | SMTP username | — |
| `ALERT_SMTP_PASS` | SMTP password | — |
| `ALERT_EMAIL_TO` | Alert recipient email | — |

---

## Tech stack

- **Python 3.9+**
- **TensorFlow 2.13 + Keras** — CNN architecture and training
- **OpenCV 4.8** — video capture, face detection, frame processing
- **FaceNet** — 128-dimensional face embedding model
- **Flask** — web server and MJPEG stream
- **SQLite** — embedded database
- **NumPy** — embedding math and triplet sampling

---

## Use cases

- Shopping mall entrance monitoring
- Airport security screening
- Residential complex access control
- Office attendance tracking

---

## Author

**Pranav Sahith Masapu**  
[LinkedIn](https://linkedin.com/in/pranav-sahith) · [GitHub](https://github.com/pranavsahtih2104)
