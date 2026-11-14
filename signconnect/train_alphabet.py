"""Train the fingerspelling (A-Z) recogniser from the still-image dataset.

    python -m signconnect.train_alphabet

Reads   <ALPHABET_IMAGE_DIR>/<LETTER>/*.jpg
Writes  models/alphabet_model.joblib
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .landmarks import Hand, HandTracker, alphabet_features, mirror_hands

# The dataset images are only 128 px, and many hands touch the image border.
# Trying a few (size, padding) variants finds noticeably more hands.
DETECT_VARIANTS = ((512, 0), (384, 64), (640, 0), (256, 96))


def detect_image_hands(image_bgr: np.ndarray, tracker: HandTracker) -> List[Hand]:
    """Detect hands in a still image; keeps the attempt that finds most hands."""
    h, w = image_bgr.shape[:2]
    best: List[Hand] = []
    for size, pad in DETECT_VARIANTS:
        scale = size / max(h, w)
        img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        if pad:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        hands = tracker.detect(img)
        if len(hands) > len(best):
            best = hands
        if len(best) == 2:
            break
    return best


def _rotate(hands: Sequence[Hand], degrees: float) -> List[Hand]:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return [Hand(h.points @ rot.T, h.label, h.score) for h in hands]


def augment(hands: Sequence[Hand], rng: np.random.Generator, copies: int) -> List[np.ndarray]:
    """Create varied copies of one sample: rotation, jitter, mirroring."""
    out = []
    for _ in range(copies):
        hs = _rotate(hands, rng.uniform(-18, 18))
        if rng.random() < 0.5:
            hs = mirror_hands(hs, 0.0)
        jittered = []
        for h in hs:
            pts = h.points + rng.normal(0, 0.025 * h.size, h.points.shape).astype(np.float32)
            pts = (pts - pts[0]) * rng.uniform(0.9, 1.1) + pts[0]
            jittered.append(Hand(pts, h.label, h.score))
        feats = alphabet_features(jittered)
        if feats is not None:
            out.append(feats)
    return out


def load_dataset(tracker: HandTracker) -> Tuple[List[List[Hand]], List[str], Dict[str, Tuple[int, int]]]:
    samples, labels, stats = [], [], {}
    root = config.ALPHABET_IMAGE_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Alphabet dataset not found: {root}")
    for letter_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in letter_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS)
        used = 0
        for f in files:
            image = cv2.imread(str(f))
            if image is None:
                continue
            hands = detect_image_hands(image, tracker)
            if hands:
                samples.append(hands)
                labels.append(letter_dir.name.upper())
                used += 1
        stats[letter_dir.name.upper()] = (used, len(files))
    return samples, labels, stats


def _fit(samples, labels, rng, copies):
    X, y = [], []
    for hands, label in zip(samples, labels):
        base = alphabet_features(hands)
        X.append(base)
        y.append(label)
        for feats in augment(hands, rng, copies):
            X.append(feats)
            y.append(label)
    model = make_pipeline(StandardScaler(),
                          CalibratedClassifierCV(SVC(C=10, gamma="scale"), ensemble=False, cv=3))
    model.fit(np.array(X), np.array(y))
    return model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--copies", type=int, default=8, help="augmented copies per image")
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = ap.parse_args(argv)

    with HandTracker(min_confidence=0.2) as tracker:
        samples, labels, stats = load_dataset(tracker)
    print("Images with a detected hand, per letter:")
    for letter, (used, total) in stats.items():
        print(f"  {letter}: {used}/{total}")
    labels_arr = np.array(labels)
    print(f"Total usable: {len(samples)} images, {len(set(labels))} letters")

    cv_accuracy = None
    if not args.no_cv:
        correct = 0
        for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(samples, labels_arr):
            model = _fit([samples[i] for i in train], labels_arr[train],
                         np.random.default_rng(0), args.copies)
            Xt = np.array([alphabet_features(samples[i]) for i in test])
            correct += int((model.predict(Xt) == labels_arr[test]).sum())
        cv_accuracy = correct / len(samples)
        total_images = sum(t for _, t in stats.values())
        print(f"(of all {total_images} images, counting undetected hands as wrong: "
              f"{correct / total_images:.1%})")
        print(f"5-fold cross-validated accuracy: {cv_accuracy:.1%}")

    model = _fit(samples, labels_arr, np.random.default_rng(0), args.copies)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": list(model.classes_), "sklearn_version": sklearn.__version__, "cv_accuracy": cv_accuracy,
                 "images_per_letter": stats}, config.ALPHABET_MODEL_PATH, compress=3)
    print(f"Saved {config.ALPHABET_MODEL_PATH}")
    os._exit(0)   # MediaPipe's teardown is noisy on some versions; exit cleanly


if __name__ == "__main__":
    main()
