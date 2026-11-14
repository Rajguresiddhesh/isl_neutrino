"""Central configuration: every path the project uses lives here.

All paths are relative to the repository root, so the project runs on any
machine without editing hard-coded '/Users/...' paths.
Override any of them with environment variables if your data lives elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ[name]).expanduser() if os.environ.get(name) else default


# --- datasets (already in the repo) -----------------------------------------
# One folder per word, each holding one or more clips of that sign.
WORD_VIDEO_DIR = _env_path("ISL_WORD_VIDEO_DIR", ROOT / "isl_sch 2" / "isl")
# One folder per letter, each holding still images of that fingerspelled letter.
ALPHABET_IMAGE_DIR = _env_path("ISL_ALPHABET_DIR", ROOT / "ISL_Dataset 6.42.04 PM")

# --- models ------------------------------------------------------------------
MODEL_DIR = _env_path("ISL_MODEL_DIR", ROOT / "models")
HAND_LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
FACE_HAAR_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
ALPHABET_MODEL_PATH = MODEL_DIR / "alphabet_model.joblib"
WORD_MODEL_PATH = MODEL_DIR / "word_model.joblib"

# --- outputs -----------------------------------------------------------------
OUTPUT_DIR = _env_path("ISL_OUTPUT_DIR", ROOT / "outputs")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --- recognition settings ----------------------------------------------------
SEQUENCE_LENGTH = 24          # every sign clip is resampled to this many frames
ANALYSIS_WIDTH = 960          # frames are downscaled to this width before detection
ANALYSIS_FPS = 12.5           # frames per second actually analysed
WORD_CONFIDENCE = 0.45        # below this a segment is reported as unknown
LETTER_CONFIDENCE = 0.50
REST_LINE = 3.6               # wrist lower than this many face-widths below the face = resting
MIN_SIGN_SECONDS = 0.30       # shorter bursts of movement are ignored
MERGE_GAP_SECONDS = 0.30      # brief tracking drop-outs inside one sign are bridged
MAX_SIGN_SECONDS = 3.5        # longer segments are assumed to hold several signs
TEMPLATE_TOLERANCE = 2.0      # reject clips more than this x normal variation away from every known sign
