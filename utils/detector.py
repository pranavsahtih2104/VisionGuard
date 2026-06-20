"""
utils/detector.py
Two-stage face detector:
  1. OpenCV DNN (ResNet SSD) — accurate, handles rotations and occlusions
  2. Haar Cascade fallback    — works without model file download
"""

import cv2
import numpy as np
import os


class FaceDetector:
    """
    Detects faces in a BGR frame.
    Returns list of (x, y, w, h) bounding boxes.
    """

    DNN_MODEL_DIR   = "models"
    PROTO_FILE      = "models/deploy.prototxt"
    WEIGHTS_FILE    = "models/res10_300x300_ssd_iter_140000.caffemodel"
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self):
        self.net      = None
        self.cascade  = None
        self._init_detector()

    def detect(self, frame: np.ndarray) -> list:
        """Returns [(x, y, w, h), ...] for all detected faces."""
        if self.net is not None:
            return self._detect_dnn(frame)
        return self._detect_cascade(frame)

    # ── DNN detector ───────────────────────────────────────────────────────────

    def _detect_dnn(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.CONFIDENCE_THRESHOLD:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            boxes.append((x1, y1, x2-x1, y2-y1))
        return boxes

    # ── Haar cascade fallback ──────────────────────────────────────────────────

    def _detect_cascade(self, frame: np.ndarray) -> list:
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray   = cv2.equalizeHist(gray)
        faces  = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [tuple(f) for f in faces] if len(faces) > 0 else []

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_detector(self):
        # Try DNN first
        if os.path.exists(self.PROTO_FILE) and os.path.exists(self.WEIGHTS_FILE):
            try:
                self.net = cv2.dnn.readNetFromCaffe(self.PROTO_FILE, self.WEIGHTS_FILE)
                print("[FaceDetector] Using OpenCV DNN (ResNet SSD)")
                return
            except Exception as e:
                print(f"[FaceDetector] DNN load failed: {e}")

        # Haar cascade fallback
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        print("[FaceDetector] Using Haar Cascade (fallback)")
