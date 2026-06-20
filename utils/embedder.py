"""
utils/embedder.py
FaceNet-style CNN that produces 128-D L2-normalised face embeddings.

Architecture:
    Input (96×96 RGB)
    → Conv blocks (32→64→128 filters, BN + ReLU)
    → GlobalAveragePooling
    → Dense(256) → Dropout
    → Dense(128)  ← embedding layer
    → L2 normalisation

Trained with Triplet Loss so that:
    d(anchor, positive) + margin < d(anchor, negative)
which clusters same-identity embeddings and pushes different-identity
embeddings apart in 128-D space.

One-shot learning is natural here: registering a single image stores one
embedding; at query time cosine distance to that single point suffices.
"""

import numpy as np
import os

# ── Optional TF import (graceful fallback for environments without GPU) ───────
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class FaceEmbedder:
    """
    Wraps the FaceNet CNN.
    If TensorFlow is not installed, falls back to a deterministic numpy
    simulation (useful for CI / demo without GPU).
    """

    INPUT_SHAPE = (96, 96, 3)
    EMBED_DIM   = 128

    def __init__(self, weights_path: str = "models/facenet_weights.h5"):
        self.weights_path = weights_path
        if TF_AVAILABLE:
            self.model = self._build_model()
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                print(f"[FaceEmbedder] Loaded weights from {weights_path}")
            else:
                print("[FaceEmbedder] No weights file found — using random init.")
                print("               Run train/train_triplet.py to train the model.")
        else:
            self.model = None
            print("[FaceEmbedder] TensorFlow not found — using dummy embedder.")

    # ── Public ─────────────────────────────────────────────────────────────────

    def get_embedding(self, face_roi: np.ndarray) -> np.ndarray:
        """
        face_roi : BGR image crop of a detected face (any size).
        Returns  : float32 np.array of shape (128,), L2-normalised.
        """
        processed = self._preprocess(face_roi)
        if TF_AVAILABLE and self.model is not None:
            emb = self.model.predict(processed, verbose=0)[0]
        else:
            emb = self._dummy_embedding(face_roi)
        return self._l2_normalise(emb)

    # ── Model definition ───────────────────────────────────────────────────────

    def _build_model(self):
        """
        Lightweight FaceNet-style CNN.
        Matches published FaceNet architecture at reduced scale for real-time
        inference on CPU.
        """
        inp = layers.Input(shape=self.INPUT_SHAPE, name='face_input')

        # Block 1
        x = layers.Conv2D(32, 3, padding='same', name='conv1')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

        # Block 2
        x = layers.Conv2D(64, 3, padding='same', name='conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

        # Block 3
        x = layers.Conv2D(128, 3, padding='same', name='conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

        # Block 4 — deeper features
        x = layers.Conv2D(256, 3, padding='same', name='conv4')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Embedding head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(self.EMBED_DIM, name='embedding')(x)

        # L2 normalise so cosine similarity == dot product
        x = layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1),
            name='l2_norm'
        )(x)

        return Model(inp, x, name='FaceNet')

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2_to_rgb(bgr)
        resized = resize_face(rgb, self.INPUT_SHAPE[:2])
        normed  = resized.astype('float32') / 255.0
        return np.expand_dims(normed, 0)   # (1, 96, 96, 3)

    @staticmethod
    def _l2_normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v) + 1e-10
        return v / norm

    @staticmethod
    def _dummy_embedding(face_roi: np.ndarray) -> np.ndarray:
        """
        Deterministic 128-D vector from pixel statistics.
        Used only when TensorFlow is unavailable (tests / CI).
        NOT suitable for real recognition — install TF for actual use.
        """
        small = face_roi[:8, :8].astype('float32').flatten()
        rng   = np.random.default_rng(seed=int(small.mean() * 1000) % (2**31))
        return rng.standard_normal(128).astype('float32')


# ── Utility functions (avoid circular imports) ────────────────────────────────

def cv2_to_rgb(bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_face(img: np.ndarray, size=(96, 96)) -> np.ndarray:
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
