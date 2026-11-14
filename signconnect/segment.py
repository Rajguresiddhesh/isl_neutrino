"""Sign segmentation: split a video into one chunk per sign.

This is what the empty `video_to_text/move_detect.py` was meant to become.
Signers return their hands to a rest position (or out of view) between signs,
so a sign is a run of frames in which at least one hand is raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from . import config
from .landmarks import BodyReference, FrameObservation, active_hands


@dataclass
class Segment:
    start: int      # index of first frame (inclusive)
    end: int        # index of last frame (exclusive)
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def find_segments(observations: Sequence[FrameObservation], ref: BodyReference) -> List[Segment]:
    if not observations:
        return []
    active = [bool(active_hands(o, ref)) for o in observations]
    times = [o.time for o in observations]
    dt = (times[-1] - times[0]) / max(len(times) - 1, 1) or 0.08

    # 1. raw runs of active frames
    runs, start = [], None
    for i, a in enumerate(active + [False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append([start, i])
            start = None

    # 2. bridge short tracking drop-outs
    merged: List[List[int]] = []
    for run in runs:
        if merged and (run[0] - merged[-1][1]) * dt <= config.MERGE_GAP_SECONDS:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    # 3. drop blips
    segments = []
    for s, e in merged:
        if (e - s) * dt >= config.MIN_SIGN_SECONDS:
            segments.append(Segment(s, e, times[s], times[e - 1] + dt))
    return segments
