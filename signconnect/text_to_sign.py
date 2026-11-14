"""Text -> sign: turn ISL glosses into one video.

Workflow (matches the project slide): fetch each sign's clip from the dataset,
fall back to fingerspelling for words that have no clip, concatenate with
OpenCV and write a browser-playable H.264 file.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from . import config
from .grammar import ISLSentence, english_to_isl

FRAME_W, FRAME_H, FPS = 960, 540, 25
LETTER_SECONDS = 0.8
PAUSE_SECONDS = 0.5           # blank beat between sentences
CAPTION_H = 64


@dataclass
class PlanItem:
    gloss: str
    kind: str                           # "video" | "fingerspell"
    clip: Optional[Path] = None
    letters: List[str] = field(default_factory=list)


@dataclass
class RenderResult:
    path: Optional[Path]
    sentences: List[ISLSentence]
    plan: List[List[PlanItem]]
    signed_from_video: List[str]
    fingerspelled: List[str]
    duration: float = 0.0


class SignLibrary:
    """Index of the sign clips and letter images that exist on disk."""

    def __init__(self, word_dir: Path = None, alphabet_dir: Path = None):
        self.word_dir = Path(word_dir or config.WORD_VIDEO_DIR)
        self.alphabet_dir = Path(alphabet_dir or config.ALPHABET_IMAGE_DIR)
        self.clips: Dict[str, Path] = {}
        self.letters: Dict[str, Path] = {}
        self.refresh()

    def refresh(self) -> None:
        self.clips.clear()
        self.letters.clear()
        if self.word_dir.is_dir():
            for d in sorted(p for p in self.word_dir.iterdir() if p.is_dir()):
                clips = sorted(f for f in d.iterdir() if f.suffix.lower() in config.VIDEO_EXTENSIONS)
                if clips:
                    self.clips[d.name.lower().strip()] = clips[0]
        if self.alphabet_dir.is_dir():
            for d in sorted(p for p in self.alphabet_dir.iterdir() if p.is_dir()):
                images = sorted((f for f in d.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS),
                                key=lambda f: [int(n) for n in re.findall(r"\d+", f.name)] or [0])
                if images and len(d.name) == 1:
                    self.letters[d.name.upper()] = images[0]

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self.clips)

    def plan(self, glosses: Sequence[str]) -> List[PlanItem]:
        items = []
        for gloss in glosses:
            key = gloss.lower().strip()
            if key in self.clips:
                items.append(PlanItem(gloss, "video", clip=self.clips[key]))
            else:
                letters = [c.upper() for c in key if c.isalnum()]
                if letters:
                    items.append(PlanItem(gloss, "fingerspell", letters=letters))
        return items


# --------------------------------------------------------------------------- #
# Frame drawing
# --------------------------------------------------------------------------- #
def _fit(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox a frame into w x h."""
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(int(fw * scale), 1), max(int(fh * scale), 1)
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _caption(frame: np.ndarray, glosses: Sequence[str], current: int, hint: Optional[str]) -> np.ndarray:
    """Bottom bar listing the gloss sentence with the current sign highlighted."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - CAPTION_H), (w, h), (20, 20, 20), -1)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    words = [g.upper() for g in glosses]
    total = sum(cv2.getTextSize(wd + " ", font, scale, thick)[0][0] for wd in words)
    if total > w - 40:
        scale *= (w - 40) / total
    x = max((w - int(sum(cv2.getTextSize(wd + " ", font, scale, thick)[0][0] for wd in words))) // 2, 20)
    for i, wd in enumerate(words):
        colour = (80, 220, 255) if i == current else (190, 190, 190)
        cv2.putText(out, wd, (x, h - 22), font, scale, colour, thick, cv2.LINE_AA)
        x += cv2.getTextSize(wd + " ", font, scale, thick)[0][0]
    if hint:
        cv2.rectangle(out, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.putText(out, hint, (16, 27), font, 0.6, (120, 230, 120), 1, cv2.LINE_AA)
    return out


def _letter_card(letter: str, image_path: Optional[Path], word: str, index: int) -> np.ndarray:
    """Still frame for one fingerspelled letter."""
    canvas = np.full((FRAME_H, FRAME_W, 3), 24, dtype=np.uint8)
    body_h = FRAME_H - CAPTION_H
    image = cv2.imread(str(image_path)) if image_path else None
    if image is not None:
        side = body_h - 110
        pic = cv2.resize(image, (side, side), interpolation=cv2.INTER_CUBIC)
        x, y = (FRAME_W - side) // 2, 50
        canvas[y:y + side, x:x + side] = pic
    else:   # no picture for this letter/digit in the dataset: show it as text
        size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 8, 14)[0]
        cv2.putText(canvas, letter, ((FRAME_W - size[0]) // 2, (body_h + size[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 8, (235, 235, 235), 14, cv2.LINE_AA)
    # the word being spelled, current letter highlighted
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 1.0
    spelled = word.upper()
    widths = [cv2.getTextSize(c, font, scale, 2)[0][0] + 6 for c in spelled]
    x = (FRAME_W - sum(widths)) // 2
    for i, c in enumerate(spelled):
        colour = (80, 220, 255) if i == index else (150, 150, 150)
        cv2.putText(canvas, c, (x, body_h - 20), font, scale, colour, 2, cv2.LINE_AA)
        x += widths[i]
    return canvas


def _clip_frames(path: Path, trim: Optional[Sequence[float]] = None):
    """Yield frames of a sign clip, optionally trimmed to [start, end] seconds."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    if not 1 <= fps <= 240:
        fps = FPS
    step, index, next_pick = fps / FPS, 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            index += 1
            if trim and (t < trim[0] or t > trim[1]):
                continue
            if step > 1 and index - 1 < next_pick - 1e-6:      # drop frames of high-fps clips
                continue
            next_pick += step
            yield _fit(frame, FRAME_W, FRAME_H)
    finally:
        cap.release()


def _load_trims() -> Dict[str, List[float]]:
    """Sign start/end times found during training, used to cut idle lead-in."""
    try:
        import joblib
        return joblib.load(config.WORD_MODEL_PATH).get("trims", {})
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def render_sentences(sentences: Sequence[ISLSentence], out_path: Path = None,
                     library: SignLibrary = None, trim_idle: bool = True) -> RenderResult:
    """Render gloss sentences into one mp4. Returns what was signed and how."""
    import imageio

    library = library or SignLibrary()
    plans = [library.plan(s.glosses) for s in sentences]
    used_video = [i.gloss for p in plans for i in p if i.kind == "video"]
    spelled = [i.gloss for p in plans for i in p if i.kind == "fingerspell"]
    if not any(plans):
        return RenderResult(None, list(sentences), plans, used_video, spelled)

    if out_path is None:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.OUTPUT_DIR / f"isl_{int(time.time() * 1000)}.mp4"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trims = _load_trims() if trim_idle else {}

    writer = imageio.get_writer(str(out_path), fps=FPS, codec="libx264", quality=7,
                                macro_block_size=2, pixelformat="yuv420p")
    frames_written = 0

    def write(frame_bgr: np.ndarray) -> None:
        nonlocal frames_written
        writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        frames_written += 1

    try:
        last = None
        for s_index, (sentence, plan) in enumerate(zip(sentences, plans)):
            shown = [item.gloss for item in plan]
            hint = sentence.facial_expression
            for g_index, item in enumerate(plan):
                if item.kind == "video":
                    trim = trims.get(item.gloss.lower())
                    for frame in _clip_frames(item.clip, trim):
                        last = _caption(frame, shown, g_index, hint)
                        write(last)
                else:
                    for l_index, letter in enumerate(item.letters):
                        card = _letter_card(letter, library.letters.get(letter), item.gloss, l_index)
                        last = _caption(card, shown, g_index, hint)
                        for _ in range(int(LETTER_SECONDS * FPS)):
                            write(last)
            if last is not None and s_index < len(sentences) - 1:
                for _ in range(int(PAUSE_SECONDS * FPS)):
                    write(last)
    finally:
        writer.close()
    return RenderResult(out_path, list(sentences), plans, used_video, spelled, frames_written / FPS)


def text_to_sign_video(text: str, out_path: Path = None, library: SignLibrary = None) -> RenderResult:
    """English text -> ISL grammar -> sign video."""
    return render_sentences(english_to_isl(text), out_path, library)
