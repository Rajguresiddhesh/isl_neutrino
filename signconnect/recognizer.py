"""Sign -> text: recognise ISL word signs in a video and letters in an image."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np

from . import config
from .landmarks import (BodyReference, FrameObservation, Hand, HandTracker, alphabet_features,
                        frame_features, observe_video, sequence_features)
from .segment import Segment, find_segments

UNKNOWN = "?"


@dataclass
class SignPrediction:
    label: str                    # recognised word, or "?" when unsure
    confidence: float             # 0..1
    t_start: float
    t_end: float
    best_guess: str = ""          # most likely word even if it was rejected
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class VideoResult:
    predictions: List[SignPrediction]
    hands_seen: bool
    face_seen: bool
    duration: float

    @property
    def glosses(self) -> List[str]:
        return [p.label for p in self.predictions if p.label != UNKNOWN]


def _load(path: Path, trainer: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model file {path.name} is missing. Create it with:  python -m signconnect.{trainer}")
    try:
        return joblib.load(path)
    except Exception as exc:  # usually a scikit-learn version mismatch
        raise RuntimeError(
            f"Could not load {path.name} ({exc}). It was probably saved with a different "
            f"scikit-learn version; rebuild it with:  python -m signconnect.{trainer}") from exc


class SignRecognizer:
    """Loads the trained models once and reuses them."""

    def __init__(self):
        self._word = None
        self._alphabet = None
        self._tracker: Optional[HandTracker] = None

    # -- lazy resources ------------------------------------------------------
    @property
    def tracker(self) -> HandTracker:
        if self._tracker is None:
            self._tracker = HandTracker()
        return self._tracker

    @property
    def word_bundle(self) -> dict:
        if self._word is None:
            self._word = _load(config.WORD_MODEL_PATH, "train_words")
        return self._word

    @property
    def alphabet_bundle(self) -> dict:
        if self._alphabet is None:
            self._alphabet = _load(config.ALPHABET_MODEL_PATH, "train_alphabet")
        return self._alphabet

    @property
    def vocabulary(self) -> List[str]:
        return list(self.word_bundle["classes"])

    @property
    def letters(self) -> List[str]:
        return list(self.alphabet_bundle["classes"])

    # -- words ---------------------------------------------------------------
    def _classify(self, frames: np.ndarray) -> Tuple[str, float, str, List[Tuple[str, float]]]:
        from .train_words import template_distance

        bundle = self.word_bundle
        feats = sequence_features(frames)
        probs = bundle["model"].predict_proba(feats[None])[0]
        order = np.argsort(probs)[::-1]
        ranked = [(str(bundle["classes"][i]), float(probs[i])) for i in order[:3]]
        best, confidence = ranked[0]

        # Is this clip actually close to any example of the winning word?
        template = bundle.get("templates", {}).get(best)
        if template:
            distance = min(template_distance(feats, t) for t in template["features"])
            closeness = template["limit"] * config.TEMPLATE_TOLERANCE / max(distance, 1e-6)
            confidence *= float(np.clip(closeness, 0.0, 1.0))
        label = best if confidence >= config.WORD_CONFIDENCE else UNKNOWN
        return label, confidence, best, ranked

    def _predict_span(self, observations: Sequence[FrameObservation], ref: BodyReference,
                      start: int, end: int, dt: float) -> SignPrediction:
        frames = np.stack([frame_features(o, ref) for o in observations[start:end]])
        label, conf, best, ranked = self._classify(frames)
        return SignPrediction(label, conf, observations[start].time,
                              observations[end - 1].time + dt, best, ranked)

    def _split_long_segment(self, observations, ref, seg: Segment, dt: float) -> List[SignPrediction]:
        """A long stretch without rest probably holds several signs: slide a
        window across it and keep each run of agreeing confident predictions."""
        window = max(int(round(1.2 / dt)), 4)
        stride = max(int(round(0.3 / dt)), 1)
        hits: List[SignPrediction] = []
        for s in range(seg.start, max(seg.end - window, seg.start) + 1, stride):
            hits.append(self._predict_span(observations, ref, s, min(s + window, seg.end), dt))
        merged: List[SignPrediction] = []
        for hit in hits:
            if hit.label == UNKNOWN:
                continue
            if merged and merged[-1].label == hit.label and hit.t_start <= merged[-1].t_end + 0.6:
                merged[-1].t_end = hit.t_end
                merged[-1].confidence = max(merged[-1].confidence, hit.confidence)
            else:
                merged.append(hit)
        return merged or [self._predict_span(observations, ref, seg.start, seg.end, dt)]

    def recognize_observations(self, observations: Sequence[FrameObservation],
                               ref: BodyReference) -> VideoResult:
        times = [o.time for o in observations]
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1) or 0.08
        predictions: List[SignPrediction] = []
        for seg in find_segments(observations, ref):
            if seg.duration > config.MAX_SIGN_SECONDS:
                predictions.extend(self._split_long_segment(observations, ref, seg, dt))
            else:
                predictions.append(self._predict_span(observations, ref, seg.start, seg.end, dt))
        return VideoResult(predictions, any(o.hands for o in observations), ref.found,
                           times[-1] + dt)

    def recognize_video(self, path: str | Path) -> VideoResult:
        observations, ref = observe_video(path, self.tracker)
        return self.recognize_observations(observations, ref)

    # -- letters -------------------------------------------------------------
    def recognize_letter(self, image_bgr: np.ndarray
                         ) -> Tuple[str, float, List[Hand], List[Tuple[str, float]]]:
        """Returns (letter or '?', confidence, hands, top-3 alternatives)."""
        from .train_alphabet import detect_image_hands

        hands = detect_image_hands(image_bgr, self.tracker)
        feats = alphabet_features(hands)
        if feats is None:
            return UNKNOWN, 0.0, [], []
        bundle = self.alphabet_bundle
        probs = bundle["model"].predict_proba(feats[None])[0]
        order = np.argsort(probs)[::-1]
        ranked = [(str(bundle["classes"][i]), float(probs[i])) for i in order[:3]]
        letter, confidence = ranked[0]
        if confidence < config.LETTER_CONFIDENCE:
            letter = UNKNOWN
        return letter, confidence, hands, ranked
