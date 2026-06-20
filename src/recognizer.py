"""
src/recognizer.py
Core recognition engine:
  - OpenCV face detection
  - CNN feature extraction via FaceNet (128-D embeddings)
  - Triplet-loss-trained similarity matching
  - One-shot learning: register with a single image
"""

import cv2
import numpy as np
import base64
import time
import os
from datetime import datetime
from utils.embedder import FaceEmbedder
from utils.detector import FaceDetector


class FaceRecognizer:
    """
    Orchestrates detection → embedding → matching → alerting.

    Workflow:
        1. Grab frame from camera (IP or webcam).
        2. Detect faces with OpenCV DNN / Haar cascade.
        3. Pass each ROI through FaceNet CNN → 128-D embedding.
        4. Compare embedding against known database using cosine distance
           (trained with triplet loss, so same-person embeddings cluster tightly).
        5. If distance < threshold → RECOGNISED (access granted).
           Else                   → UNKNOWN    (alert + store face).
    """

    RECOGNITION_THRESHOLD = 0.55   # cosine distance; lower = stricter
    FRAME_SKIP            = 2      # process every N-th frame for speed
    UNKNOWN_SAVE_DIR      = "database/unknown_faces"

    def __init__(self, db):
        self.db        = db
        self.detector  = FaceDetector()
        self.embedder  = FaceEmbedder()
        self._frame_count = 0
        self._last_result = {}          # cache last annotation between skipped frames
        os.makedirs(self.UNKNOWN_SAVE_DIR, exist_ok=True)

        # Load existing embeddings from DB into memory for fast lookup
        self._known_embeddings = {}     # { person_id: np.array(128,) }
        self._load_known_faces()

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_frames(self):
        """Yield MJPEG-encoded frames with bounding boxes and labels."""
        cap = self._open_camera()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                self._frame_count += 1
                if self._frame_count % self.FRAME_SKIP == 0:
                    self._last_result = self._process_frame(frame)

                annotated = self._annotate(frame, self._last_result)
                _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
        finally:
            cap.release()

    def register_person(self, name: str, image_b64: str) -> dict:
        """
        One-shot registration: a single image is enough.
        Decodes base64 image → detects face → extracts embedding → stores in DB.
        """
        img = self._decode_b64(image_b64)
        if img is None:
            return {'success': False, 'error': 'Invalid image data'}

        faces = self.detector.detect(img)
        if not faces:
            return {'success': False, 'error': 'No face detected in image'}

        x, y, w, h = faces[0]
        roi = img[y:y+h, x:x+w]
        embedding = self.embedder.get_embedding(roi)

        person_id = self.db.register_person(name, embedding.tolist())
        self._known_embeddings[person_id] = embedding
        return {'success': True, 'person_id': person_id, 'name': name}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> dict:
        """Detect faces, embed, match. Returns annotation dict."""
        results = []
        faces = self.detector.detect(frame)

        for (x, y, w, h) in faces:
            roi       = frame[y:y+h, x:x+w]
            embedding = self.embedder.get_embedding(roi)
            name, dist, pid = self._match(embedding)

            recognised = dist < self.RECOGNITION_THRESHOLD
            label      = name if recognised else "Unknown"
            color      = (34, 197, 94) if recognised else (239, 68, 68)  # green / red

            results.append({
                'box':   (x, y, w, h),
                'label': label,
                'dist':  round(float(dist), 3),
                'color': color,
                'known': recognised,
            })

            # Log & alert on unknown faces
            if not recognised:
                self._handle_unknown(frame, roi, embedding, x, y, w, h)
            else:
                self.db.log_recognition(pid, dist)

        return {'faces': results, 'timestamp': time.time()}

    def _match(self, embedding: np.ndarray):
        """
        Cosine distance against all known embeddings.
        Triplet-loss training ensures intra-class distance << inter-class distance.
        Returns (name, distance, person_id).
        """
        best_dist = float('inf')
        best_pid  = None

        for pid, known_emb in self._known_embeddings.items():
            dist = self._cosine_distance(embedding, known_emb)
            if dist < best_dist:
                best_dist = dist
                best_pid  = pid

        if best_pid is None:
            return "Unknown", 1.0, None

        name = self.db.get_person_name(best_pid)
        return name, best_dist, best_pid

    def _handle_unknown(self, frame, roi, embedding, x, y, w, h):
        """Save unknown face image + trigger alert."""
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.UNKNOWN_SAVE_DIR}/unknown_{ts}.jpg"
        cv2.imwrite(filename, roi)
        self.db.log_alert(filename, embedding.tolist())

    def _annotate(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        out = frame.copy()
        for f in result.get('faces', []):
            x, y, w, h  = f['box']
            label        = f['label']
            color        = f['color']
            dist_str     = f"dist:{f['dist']:.2f}"

            # Box
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x, y-lh-10), (x+lw+6, y), color, -1)
            cv2.putText(out, label, (x+3, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Distance
            cv2.putText(out, dist_str, (x, y+h+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Overlay timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(out, ts, (10, out.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        return out

    def _load_known_faces(self):
        rows = self.db.get_all_embeddings()
        for row in rows:
            self._known_embeddings[row['id']] = np.array(row['embedding'])

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return float(1.0 - np.dot(a, b))

    @staticmethod
    def _decode_b64(b64_str: str):
        try:
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            data = base64.b64decode(b64_str)
            arr  = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _open_camera():
        # Try webcam first; fall back to IP cam env var
        ip_cam = os.getenv('IP_CAMERA_URL')
        if ip_cam:
            cap = cv2.VideoCapture(ip_cam)
        else:
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
