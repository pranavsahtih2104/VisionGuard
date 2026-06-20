"""
train/train_triplet.py
Trains the FaceNet CNN with Triplet Loss.

Triplet Loss:
    L = max(0, d(a,p) - d(a,n) + margin)

    a = anchor   (a face of person X)
    p = positive (another face of same person X)
    n = negative (a face of different person Y)

    Forces: same-person embeddings cluster together,
            different-person embeddings push apart.

Usage:
    python train/train_triplet.py --data_dir data/faces --epochs 30

Dataset structure expected:
    data/faces/
        PersonA/
            img1.jpg
            img2.jpg
        PersonB/
            img1.jpg
        ...
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from utils.embedder import FaceEmbedder
    TF_OK = True
except ImportError:
    TF_OK = False
    print("TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)


# ── Triplet Loss ───────────────────────────────────────────────────────────────

def triplet_loss(margin=0.3):
    def loss(y_true, y_pred):
        """
        y_pred shape: (batch, 3 * 128) — [anchor, positive, negative] concatenated.
        """
        emb_size = 128
        anchor   = y_pred[:, :emb_size]
        positive = y_pred[:, emb_size:2*emb_size]
        negative = y_pred[:, 2*emb_size:]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, img_size=(96, 96)):
    """Returns dict { label: [np.array(96,96,3), ...] }"""
    import cv2
    dataset = {}
    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        imgs = []
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img  = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            imgs.append(img)
        if imgs:
            dataset[person] = imgs
    return dataset


def generate_triplets(dataset: dict, n_triplets=5000):
    """Randomly sample (anchor, positive, negative) triplets."""
    people  = list(dataset.keys())
    anchors, positives, negatives = [], [], []

    for _ in range(n_triplets):
        # Pick a person with at least 2 images for anchor + positive
        pos_person = np.random.choice([p for p in people if len(dataset[p]) >= 2])
        neg_person = np.random.choice([p for p in people if p != pos_person])

        a_idx, p_idx = np.random.choice(len(dataset[pos_person]), 2, replace=False)
        n_idx        = np.random.randint(len(dataset[neg_person]))

        anchors.append(dataset[pos_person][a_idx])
        positives.append(dataset[pos_person][p_idx])
        negatives.append(dataset[neg_person][n_idx])

    return (np.array(anchors),
            np.array(positives),
            np.array(negatives))


# ── Triplet model wrapper ──────────────────────────────────────────────────────

def build_triplet_model(base_model):
    """Wraps base FaceNet model into a triplet-input model."""
    from tensorflow.keras import Input
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras import Model

    inp_a = Input(shape=(96, 96, 3), name='anchor')
    inp_p = Input(shape=(96, 96, 3), name='positive')
    inp_n = Input(shape=(96, 96, 3), name='negative')

    emb_a = base_model(inp_a)
    emb_p = base_model(inp_p)
    emb_n = base_model(inp_n)

    merged = concatenate([emb_a, emb_p, emb_n], axis=1, name='triplet_output')
    return Model(inputs=[inp_a, inp_p, inp_n], outputs=merged, name='TripletNet')


# ── Main training loop ─────────────────────────────────────────────────────────

def train(data_dir: str, epochs: int = 30, batch_size: int = 32,
          margin: float = 0.3, lr: float = 1e-4):

    print(f"Loading dataset from: {data_dir}")
    dataset = load_dataset(data_dir)
    if len(dataset) < 2:
        print("Need at least 2 people in dataset. Exiting.")
        return

    print(f"Found {len(dataset)} identities.")
    anchors, positives, negatives = generate_triplets(dataset)
    dummy_labels = np.zeros((len(anchors), 384))   # unused by triplet loss

    embedder = FaceEmbedder(weights_path="models/facenet_weights.h5")
    base_model = embedder.model
    triplet_model = build_triplet_model(base_model)

    triplet_model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=triplet_loss(margin=margin)
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/facenet_weights.h5",
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    print(f"\nTraining for {epochs} epochs...")
    triplet_model.fit(
        [anchors, positives, negatives], dummy_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
    )

    base_model.save_weights("models/facenet_weights.h5")
    print("Training complete. Weights saved to models/facenet_weights.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VisionGuard FaceNet with Triplet Loss')
    parser.add_argument('--data_dir',   default='data/faces',  help='Path to face dataset')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--margin',     type=float, default=0.3)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.margin, args.lr)
