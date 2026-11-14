"""Live sign -> text from a webcam (or any video source).

    python -m signconnect.live                 # default camera
    python -m signconnect.live --source 1      # another camera
    python -m signconnect.live --source clip.mp4 --no-display

Keys:  q quit   c clear sentence   l toggle fingerspelling (letter) mode
       SPACE (letter mode) add the shown letter   BACKSPACE delete last letter

Word mode: sign, then lower your hands (or take them out of view); the sign is
recognised when the hands come to rest.
"""
from __future__ import annotations

import argparse
import time
from typing import List

import cv2

from . import config
from .gloss_to_english import glosses_to_english
from .landmarks import SignerObserver, active_hands, draw_hands
from .recognizer import UNKNOWN, SignRecognizer


class LiveSession:
    """Incremental recogniser: feed frames, collect recognised signs."""

    def __init__(self, recognizer: SignRecognizer):
        self.recognizer = recognizer
        self.observer = SignerObserver(recognizer.tracker)
        self.buffer = []                 # observations of the sign in progress
        self.idle_since = None
        self.glosses: List[str] = []
        self.last = ""

    def feed(self, t: float, frame):
        obs = self.observer.observe(t, frame)
        ref = self.observer.reference((obs.width, obs.height))
        if active_hands(obs, ref):
            self.buffer.append(obs)
            self.idle_since = None
        elif self.buffer:
            self.buffer.append(obs)
            self.idle_since = self.idle_since if self.idle_since is not None else t
            if t - self.idle_since > config.MERGE_GAP_SECONDS:
                self._finish(ref)
        return obs

    def _finish(self, ref) -> None:
        observations, self.buffer, self.idle_since = self.buffer, [], None
        result = self.recognizer.recognize_observations(observations, ref)
        for p in result.predictions:
            self.last = f"{p.best_guess} ({p.confidence:.0%})"
            if p.label != UNKNOWN:
                self.glosses.append(p.label)

    def flush(self) -> None:
        if self.buffer:
            last = self.buffer[-1]
            self._finish(self.observer.reference((last.width, last.height)))

    @property
    def sentence(self) -> str:
        return glosses_to_english(self.glosses)


def _overlay(frame, lines):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30 * len(lines) - 14), (w, h), (20, 20, 20), -1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (12, h - 30 * (len(lines) - i) + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 2, cv2.LINE_AA)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Live ISL sign -> text")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--no-display", action="store_true", help="run without a window")
    args = ap.parse_args(argv)

    is_camera = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source {args.source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1 <= fps <= 240:
        fps = 25.0
    step = 1 if is_camera else max(int(round(fps / config.ANALYSIS_FPS)), 1)

    recognizer = SignRecognizer()
    session = LiveSession(recognizer)
    letter_mode, spelled, letter = False, "", UNKNOWN
    start, index = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % step:
            continue
        t = time.time() - start if is_camera else index / fps
        if letter_mode:
            letter, conf, hands, _ = recognizer.recognize_letter(frame)
            lines = [(f"LETTER MODE  seen: {letter} ({conf:.0%})   SPACE = add", (80, 220, 255)),
                     (f"Spelled: {spelled}", (235, 235, 235))]
            shown = frame
        else:
            obs = session.feed(t, frame)
            shown = cv2.resize(frame, (obs.width, obs.height))
            shown = draw_hands(shown, obs.hands)
            lines = [(f"Signs: {' '.join(g.upper() for g in session.glosses)}   last: {session.last}",
                      (80, 220, 255)),
                     (f"English: {session.sentence}", (235, 235, 235))]
        if args.no_display:
            continue
        _overlay(shown, lines)
        cv2.imshow("SignConnect - live (q quit, c clear, l letters)", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            session.glosses.clear()
            spelled = ""
        if key == ord("l"):
            letter_mode = not letter_mode
        if letter_mode and key == 32 and letter != UNKNOWN:
            spelled += letter
        if letter_mode and key in (8, 127):
            spelled = spelled[:-1]

    session.flush()
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print("Signs:  ", " ".join(g.upper() for g in session.glosses))
    print("English:", session.sentence)
    if spelled:
        print("Spelled:", spelled)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
