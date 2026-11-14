"""Hand-landmark extraction and feature engineering.

Why landmarks instead of raw pixels?  The repo's earlier CNN learned from whole
224x224 frames, so it also learned the signer's shirt, the wall and the
lighting.  With only a handful of clips per sign that cannot generalise.
MediaPipe gives us 21 keypoints per hand; a model trained on those sees only
hand shape and movement, needs far less data and runs on a CPU.
"""
from __future__ import annotations

import atexit
import os
import urllib.request
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from . import config

HAND_FEATS = 42 + 2 + 1          # shape (21 x,y) + position (x,y) + present flag
FRAME_FEATS = 2 * HAND_FEATS     # two hand slots per frame
ALPHABET_FEATS = 42 + 42 + 2 + 1 + 1


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Hand:
    """One detected hand. `points` are (21, 2) pixel coordinates."""
    points: np.ndarray
    label: str = "Right"          # MediaPipe handedness label
    score: float = 1.0

    @property
    def wrist(self) -> np.ndarray:
        return self.points[0]

    @property
    def size(self) -> float:
        """Scale of the hand: longest wrist-to-keypoint distance."""
        return float(np.linalg.norm(self.points - self.points[0], axis=1).max()) + 1e-6

    @property
    def centre(self) -> np.ndarray:
        return self.points.mean(axis=0)


@dataclass
class FrameObservation:
    """Everything we keep from one analysed video frame."""
    time: float                              # seconds from start of video
    hands: List[Hand] = field(default_factory=list)
    width: int = 0
    height: int = 0


@dataclass
class BodyReference:
    """Where the signer is and how big they appear (from the face box)."""
    cx: float
    cy: float
    scale: float                             # face width in pixels
    found: bool = True


# --------------------------------------------------------------------------- #
# MediaPipe wrapper
# --------------------------------------------------------------------------- #
def ensure_hand_model() -> Path:
    """Return the path of the hand-landmarker model, downloading if missing."""
    path = config.HAND_LANDMARKER_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {path} ...")
        urllib.request.urlretrieve(config.HAND_LANDMARKER_URL, path)
    return path


# MediaPipe prints a traceback if a landmarker is still open while Python shuts
# down, so every tracker is closed explicitly at exit.
_open_trackers: "weakref.WeakSet" = weakref.WeakSet()


@atexit.register
def _close_all_trackers() -> None:
    for tracker in list(_open_trackers):
        tracker.close()


class HandTracker:
    """Thin wrapper over MediaPipe's HandLandmarker (Tasks API).

    The Tasks API is used because the old `mp.solutions.holistic` API that the
    original scripts imported has been removed from current MediaPipe releases.
    """

    def __init__(self, min_confidence: float = 0.3, max_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_hand_model())),
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        _open_trackers.add(self)

    def detect(self, frame_bgr: np.ndarray) -> List[Hand]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
            hands.append(Hand(pts, handed[0].category_name, float(handed[0].score)))
        return hands

    def close(self) -> None:
        if self._landmarker is None:
            return
        try:
            self._landmarker.close()
        except Exception:
            pass
        self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Face / body reference
# --------------------------------------------------------------------------- #
_face_detector = None
_face_kind = None


def _load_face_detector():
    """YuNet (bundled ONNX) if this OpenCV supports it, otherwise a Haar cascade."""
    global _face_detector, _face_kind
    if _face_kind is not None:
        return
    _face_kind = "none"
    try:
        _face_detector = cv2.FaceDetectorYN.create(str(config.FACE_YUNET_PATH), "", (320, 320), 0.6)
        _face_kind = "yunet"
        return
    except Exception:
        pass
    try:
        cascade = cv2.CascadeClassifier(str(config.FACE_HAAR_PATH))
        if not cascade.empty():
            _face_detector, _face_kind = cascade, "haar"
    except Exception:
        pass


def _detect_face(frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return (centre_x, centre_y, width) of the largest face, or None."""
    _load_face_detector()
    if _face_kind == "yunet":
        h, w = frame_bgr.shape[:2]
        _face_detector.setInputSize((w, h))
        faces = _face_detector.detect(frame_bgr)[1]
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
        return float(x + fw / 2.0), float(y + fh / 2.0), float(fw)
    if _face_kind == "haar":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = _face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                                minSize=(24, 24))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return x + fw / 2.0, y + fh / 2.0, float(fw)
    return None


# --------------------------------------------------------------------------- #
# Video reading
# --------------------------------------------------------------------------- #
def _resize(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def iter_video_frames(path: str | Path, fps: float = config.ANALYSIS_FPS,
                      ) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (time_seconds, full-resolution frame) at roughly `fps`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1.0 <= src_fps <= 240.0:
        src_fps = 25.0
    step = max(src_fps / fps, 1.0)
    index, next_pick = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= next_pick - 1e-6:
                next_pick += step
                yield index / src_fps, frame
            index += 1
    finally:
        cap.release()


def signer_region(face: Tuple[float, float, float], width: int, height: int
                  ) -> Optional[Tuple[int, int, int, int]]:
    """Box (x0, y0, x1, y1) around the signer's upper body, from the face box.

    When the signer stands far from the camera their hands are only a few
    pixels wide and palm detection fails.  Cropping to this box first lets the
    detector work at a useful resolution.  Returns None if the signer already
    fills most of the frame.
    """
    cx, cy, fw = face
    x0, x1 = int(max(cx - 4.5 * fw, 0)), int(min(cx + 4.5 * fw, width))
    y0, y1 = int(max(cy - 2.5 * fw, 0)), int(min(cy + 6.5 * fw, height))
    if (x1 - x0) * (y1 - y0) > 0.7 * width * height or x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


class SignerObserver:
    """Feeds frames one at a time and returns hand observations.

    Used for both whole videos and live camera streams.  All coordinates are
    reported in the 'analysis' frame (the input downscaled to ANALYSIS_WIDTH).
    """

    FACE_EVERY = 12       # re-locate the face every N analysed frames

    def __init__(self, tracker: HandTracker):
        self.tracker = tracker
        self.faces: List[Tuple[float, float, float]] = []   # in analysis coordinates
        self._count = 0
        self._region = None
        self._scale = 1.0

    def observe(self, time_s: float, frame: np.ndarray) -> FrameObservation:
        small = _resize(frame, config.ANALYSIS_WIDTH)
        h, w = small.shape[:2]
        self._scale = w / frame.shape[1]
        if self._count % self.FACE_EVERY == 0:
            face = _detect_face(small)
            if face is not None:
                self.faces.append(face)
                full = tuple(v / self._scale for v in face)
                self._region = signer_region(full, frame.shape[1], frame.shape[0])
        self._count += 1

        if self._region is None:
            hands = self.tracker.detect(small)
        else:
            x0, y0, x1, y1 = self._region
            crop = frame[y0:y1, x0:x1]
            zoom = min(config.ANALYSIS_WIDTH / crop.shape[1], 3.0)
            if abs(zoom - 1.0) > 0.05:
                crop = cv2.resize(crop, None, fx=zoom, fy=zoom,
                                  interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR)
            else:
                zoom = 1.0
            hands = self.tracker.detect(crop)
            for hand in hands:       # crop coordinates -> analysis coordinates
                hand.points = (hand.points / zoom + (x0, y0)) * self._scale
                hand.points = hand.points.astype(np.float32)
        return FrameObservation(time_s, hands, w, h)

    def reference(self, frame_size: Tuple[int, int]) -> BodyReference:
        """Median face position seen so far; frame centre if no face was found."""
        if self.faces:
            cx, cy, s = np.median(np.array(self.faces), axis=0)
            return BodyReference(float(cx), float(cy), float(s), True)
        w, h = frame_size
        return BodyReference(w / 2.0, h * 0.3, h * 0.12, False)


def observe_video(path: str | Path, tracker: Optional[HandTracker] = None,
                  ) -> Tuple[List[FrameObservation], BodyReference]:
    """Run hand tracking over a whole video."""
    own = tracker is None
    tracker = tracker or HandTracker()
    observer = SignerObserver(tracker)
    observations: List[FrameObservation] = []
    try:
        for t, frame in iter_video_frames(path):
            observations.append(observer.observe(t, frame))
    finally:
        if own:
            tracker.close()
    if not observations:
        raise ValueError(f"No frames could be read from {path}")
    return observations, observer.reference((observations[0].width, observations[0].height))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def hand_shape(hand: Hand) -> np.ndarray:
    """Translation- and scale-invariant hand shape, 42 numbers."""
    return ((hand.points - hand.wrist) / hand.size).reshape(-1)


def active_hands(obs: FrameObservation, ref: BodyReference) -> List[Hand]:
    """Hands that are raised into the signing space.

    Hands hanging at the signer's side (or resting on a lap) carry no meaning
    and their handedness labels flicker, so they are ignored.  A hand counts as
    raised when its wrist is less than REST_LINE face-widths below the face.
    Without a visible face every detected hand is treated as active.
    """
    if not ref.found:
        return list(obs.hands)
    return [h for h in obs.hands if (h.wrist[1] - ref.cy) / ref.scale < config.REST_LINE]


def frame_features(obs: FrameObservation, ref: BodyReference) -> np.ndarray:
    """Per-frame vector used by the word model (length FRAME_FEATS).

    Slot 0 holds the only raised hand, or the image-left one when two are
    raised; slot 1 holds the second.  MediaPipe's left/right labels are not
    used because they are unstable when the signer is far from the camera.
    """
    out = np.zeros(FRAME_FEATS, dtype=np.float32)
    hands = sorted(active_hands(obs, ref), key=lambda h: -h.score)[:2]
    hands.sort(key=lambda h: h.wrist[0])
    for slot, hand in enumerate(hands):
        o = slot * HAND_FEATS
        out[o:o + 42] = hand_shape(hand)
        out[o + 42:o + 44] = (hand.wrist - (ref.cx, ref.cy)) / ref.scale
        out[o + 44] = 1.0
    return out


def mirror_frame_features(frames: np.ndarray) -> np.ndarray:
    """Mirror a (T, FRAME_FEATS) sequence left-right (left-handed signer)."""
    out = frames.copy()
    for slot in range(2):
        o = slot * HAND_FEATS
        out[:, o:o + 42:2] *= -1.0          # x of every shape point
        out[:, o + 42] *= -1.0              # x of wrist position
    both = (out[:, 44] > 0.5) & (out[:, HAND_FEATS + 44] > 0.5)
    swapped = out[both].copy()
    out[both, :HAND_FEATS] = swapped[:, HAND_FEATS:]
    out[both, HAND_FEATS:] = swapped[:, :HAND_FEATS]
    return out


def resample_sequence(frames: np.ndarray, length: int = config.SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly resample a (T, F) sequence to (length, F)."""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 1:
        return np.repeat(frames, length, axis=0)
    src = np.linspace(0.0, 1.0, len(frames))
    dst = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(dst, src, frames[:, j]) for j in range(frames.shape[1])], axis=1
                    ).astype(np.float32)


def sequence_features(frames: np.ndarray) -> np.ndarray:
    """Flatten a clip into one vector: resampled frames + motion summary."""
    seq = resample_sequence(frames)
    velocity = np.diff(seq, axis=0)
    summary = np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0),
                              np.abs(velocity).mean(0)])
    return np.concatenate([seq.reshape(-1), summary]).astype(np.float32)


def alphabet_features(hands: Sequence[Hand]) -> Optional[np.ndarray]:
    """Single-image vector used by the fingerspelling model.

    Hands are ordered left-to-right in the image (ISL letters are mostly
    two-handed and handedness labels are unreliable on cropped hands).
    """
    hands = sorted(sorted(hands, key=lambda h: -h.score)[:2], key=lambda h: h.centre[0])
    if not hands:
        return None
    out = np.zeros(ALPHABET_FEATS, dtype=np.float32)
    a = hands[0]
    out[0:42] = hand_shape(a)
    if len(hands) == 2:
        b = hands[1]
        out[42:84] = hand_shape(b)
        out[84:86] = (b.wrist - a.wrist) / a.size
        out[86] = np.log(b.size / a.size)
        out[87] = 1.0
    return out


def mirror_hands(hands: Sequence[Hand], width: float) -> List[Hand]:
    """Horizontally flip hands (used for left-handed augmentation)."""
    flipped = []
    for h in hands:
        pts = h.points.copy()
        pts[:, 0] = width - pts[:, 0]
        flipped.append(Hand(pts, "Left" if h.label == "Right" else "Right", h.score))
    return flipped


def draw_hands(frame_bgr: np.ndarray, hands: Sequence[Hand]) -> np.ndarray:
    """Draw the hand skeletons on a frame (for the UI / debugging)."""
    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
             (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
             (18, 19), (19, 20), (0, 17)]
    out = frame_bgr.copy()
    for hand in hands:
        pts = hand.points.astype(int)
        for i, j in bones:
            cv2.line(out, tuple(pts[i]), tuple(pts[j]), (60, 200, 60), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, tuple(p), 3, (40, 40, 230), -1, cv2.LINE_AA)
    return out
