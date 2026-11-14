"""Train the word-sign recogniser from the clip dataset.

    python -m signconnect.train_words

Reads   <WORD_VIDEO_DIR>/<word>/*.mov|mp4   (one folder per sign)
Writes  models/word_model.joblib

To teach the system a new sign, add a folder named after the word containing
clips of it (different signers, several takes each) and run this again.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier

from . import config
from .landmarks import (FRAME_FEATS, HAND_FEATS, HandTracker, frame_features,
                        mirror_frame_features, observe_video, sequence_features)
from .segment import find_segments


def clip_sequence(path, tracker: HandTracker):
    """Features of the longest sign segment in a clip, and when it happens.

    Returns ((T, FRAME_FEATS) array, (start_seconds, end_seconds)) or None.
    """
    observations, ref = observe_video(path, tracker)
    segments = find_segments(observations, ref)
    if not segments:
        return None
    seg = max(segments, key=lambda s: s.duration)
    frames = np.stack([frame_features(o, ref) for o in observations[seg.start:seg.end]])
    return frames, (seg.t_start, seg.t_end)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator,
                     allow_mirror: bool = True) -> np.ndarray:
    """One randomly perturbed copy of a clip: timing, position, shape, drop-outs."""
    n = len(seq)
    # trim / shift timing
    a = int(rng.uniform(0, 0.2) * n)
    b = n - int(rng.uniform(0, 0.2) * n)
    out = seq[a:max(b, a + 2)].copy()
    if allow_mirror and rng.random() < 0.3:
        out = mirror_frame_features(out)
    angle = np.radians(rng.uniform(-12, 12))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                   dtype=np.float32)
    pos_scale = rng.uniform(0.85, 1.15)
    pos_shift = rng.normal(0, 0.3, 2)
    for slot in range(2):
        o = slot * HAND_FEATS
        present = out[:, o + 44] > 0.5
        if not present.any():
            continue
        shape = out[present, o:o + 42].reshape(-1, 21, 2) @ rot.T
        shape += rng.normal(0, 0.03, shape.shape)
        out[present, o:o + 42] = shape.reshape(-1, 42)
        pos = out[present, o + 42:o + 44] * pos_scale + pos_shift
        out[present, o + 42:o + 44] = pos + rng.normal(0, 0.06, pos.shape)
        # simulate frames where tracking lost the hand
        lost = present & (rng.random(len(out)) < 0.08)
        out[lost, o:o + HAND_FEATS] = 0.0
    return out


def build_training_set(sequences: List[np.ndarray], labels: List[str], copies: int,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for seq, label in zip(sequences, labels):
        X.append(sequence_features(seq))
        y.append(label)
        for _ in range(copies):
            X.append(sequence_features(augment_sequence(seq, rng)))
            y.append(label)
    return np.array(X), np.array(y)


def fit_model(X: np.ndarray, y: np.ndarray) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(n_estimators=200, max_depth=14, min_samples_leaf=2,
                                 random_state=0, n_jobs=-1)
    model.fit(X, y)
    return model


def template_distance(features: np.ndarray, template: np.ndarray) -> float:
    """RMS difference between two clips' resampled frame sequences."""
    n = config.SEQUENCE_LENGTH * FRAME_FEATS
    return float(np.sqrt(np.mean((features[:n] - template[:n]) ** 2)))


def build_templates(sequences: List[np.ndarray], labels: List[str], seed: int = 1) -> Dict[str, dict]:
    """Per-word reference clips plus how far a normal variation strays from them.

    A classifier always picks one of the words it knows.  Comparing a new clip
    with these templates lets the recogniser say "unknown sign" instead.
    """
    rng = np.random.default_rng(seed)
    templates: Dict[str, dict] = {}
    for seq, label in zip(sequences, labels):
        feats = sequence_features(seq)
        spread = [template_distance(sequence_features(augment_sequence(seq, rng, False)), feats)
                  for _ in range(60)]
        entry = templates.setdefault(label, {"features": [], "limit": 0.0})
        entry["features"].append(feats)
        entry["features"].append(sequence_features(mirror_frame_features(seq)))  # left-handed
        entry["limit"] = max(entry["limit"], float(np.percentile(spread, 98)))
    return templates


def _file_digest(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(tracker: HandTracker):
    """Returns sequences, labels, a per-word report and per-word trim times."""
    root = config.WORD_VIDEO_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Word video dataset not found: {root}")
    sequences, labels, report, trims = [], [], {}, {}
    for word_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        clips = sorted(f for f in word_dir.iterdir() if f.suffix.lower() in config.VIDEO_EXTENSIONS)
        seen, usable = set(), 0
        for clip in clips:
            digest = _file_digest(clip)
            if digest in seen:          # byte-identical copies add no information
                continue
            seen.add(digest)
            found = clip_sequence(clip, tracker)
            if found is None:
                print(f"  ! no sign found in {clip}")
                continue
            seq, (t0, t1) = found
            # the text->sign renderer plays the first clip of each word; remember
            # where its sign starts and ends so idle standing can be cut out
            trims.setdefault(word_dir.name.lower(), [max(t0 - 0.35, 0.0), t1 + 0.35])
            sequences.append(seq)
            labels.append(word_dir.name.lower())
            usable += 1
        report[word_dir.name.lower()] = {"clips": len(clips), "unique": len(seen), "usable": usable}
    return sequences, labels, report, trims


def leave_one_out(sequences, labels, copies) -> float | None:
    """Honest accuracy estimate; only possible with >= 2 distinct clips per word."""
    counts = defaultdict(int)
    for label in labels:
        counts[label] += 1
    if min(counts.values()) < 2:
        return None
    correct = 0
    for i in range(len(sequences)):
        train = [j for j in range(len(sequences)) if j != i]
        X, y = build_training_set([sequences[j] for j in train], [labels[j] for j in train], copies)
        model = fit_model(X, y)
        correct += int(model.predict(sequence_features(sequences[i])[None])[0] == labels[i])
    return correct / len(sequences)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--copies", type=int, default=150, help="augmented copies per clip")
    args = ap.parse_args(argv)

    with HandTracker() as tracker:
        sequences, labels, report, trims = load_dataset(tracker)
    if len(set(labels)) < 2:
        raise SystemExit("Need clips for at least two different words to train.")

    print("Clips per word (files / distinct / usable):")
    for word, r in report.items():
        print(f"  {word:12s} {r['clips']} / {r['unique']} / {r['usable']}")
    accuracy = leave_one_out(sequences, labels, max(args.copies // 3, 20))
    if accuracy is None:
        print("Accuracy cannot be measured: at least one word has only a single distinct clip, "
              "so there is nothing to hold out. Record more takes and signers.")
    else:
        print(f"Leave-one-clip-out accuracy: {accuracy:.1%}")

    X, y = build_training_set(sequences, labels, args.copies)
    model = fit_model(X, y)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": list(model.classes_), "report": report,
                 "templates": build_templates(sequences, labels), "trims": trims,
                 "sklearn_version": sklearn.__version__,
                 "loo_accuracy": accuracy, "frame_feats": FRAME_FEATS,
                 "sequence_length": config.SEQUENCE_LENGTH},
                config.WORD_MODEL_PATH, compress=3)
    print(f"Saved {config.WORD_MODEL_PATH}  ({len(model.classes_)} signs: {', '.join(model.classes_)})")
    os._exit(0)


if __name__ == "__main__":
    main()
