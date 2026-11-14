"""Command line interface.

    python -m signconnect text "She is a good student."   # English -> ISL video
    python -m signconnect gloss "What is your name?"      # English -> ISL gloss only
    python -m signconnect video path/to/signing.mp4       # sign video -> English
    python -m signconnect letter path/to/hand.jpg         # fingerspelled letter
    python -m signconnect info                            # what the system knows
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="signconnect", description="Two-way ISL interpreter")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("text", help="English text -> ISL sign video")
    p.add_argument("text")
    p.add_argument("-o", "--output", default=None)
    p = sub.add_parser("gloss", help="English text -> ISL gloss")
    p.add_argument("text")
    p = sub.add_parser("video", help="sign video -> English text")
    p.add_argument("path")
    p.add_argument("--question", action="store_true", help="signer used a questioning face")
    p = sub.add_parser("letter", help="image -> fingerspelled letter")
    p.add_argument("path")
    sub.add_parser("info", help="list known signs and letters")
    args = ap.parse_args(argv)

    if args.command == "gloss":
        from .grammar import english_to_isl
        for s in english_to_isl(args.text):
            print(f"{s.source}  ->  {s.text}")
            if s.facial_expression:
                print(f"    ({s.facial_expression})")

    elif args.command == "text":
        from .text_to_sign import text_to_sign_video
        result = text_to_sign_video(args.text, args.output)
        for s in result.sentences:
            print(f"{s.source}  ->  {s.text}")
        if result.path is None:
            print("Nothing to sign.")
            return 1
        print(f"Signed from clips : {', '.join(result.signed_from_video) or '-'}")
        print(f"Fingerspelled     : {', '.join(result.fingerspelled) or '-'}")
        print(f"Video ({result.duration:.1f}s): {result.path}")

    elif args.command == "video":
        from .gloss_to_english import glosses_to_english
        from .recognizer import SignRecognizer
        result = SignRecognizer().recognize_video(args.path)
        for p in result.predictions:
            print(f"  {p.t_start:5.1f}-{p.t_end:5.1f}s  {p.label.upper():10s} {p.confidence:.0%}"
                  + (f"  (closest: {p.best_guess})" if p.label == "?" else ""))
        if not result.hands_seen:
            print("No hands were detected in this video.")
        print("Signs:  ", " ".join(g.upper() for g in result.glosses) or "-")
        print("English:", glosses_to_english(result.glosses, args.question) or "-")
        sys.stdout.flush()
        os._exit(0)

    elif args.command == "letter":
        import cv2
        from .recognizer import SignRecognizer
        image = cv2.imread(args.path)
        if image is None:
            print(f"Cannot read image {args.path}")
            return 1
        letter, conf, hands, ranked = SignRecognizer().recognize_letter(image)
        print(f"Letter: {letter}  ({conf:.0%}; hands found: {len(hands)}; "
              f"top guesses: {', '.join(f'{l} {c:.0%}' for l, c in ranked)})")
        sys.stdout.flush()
        os._exit(0)

    elif args.command == "info":
        from .text_to_sign import SignLibrary
        lib = SignLibrary()
        print(f"Sign clips ({len(lib.clips)}): {', '.join(lib.vocabulary)}")
        print(f"Letter images ({len(lib.letters)}): {' '.join(sorted(lib.letters))}")
        missing = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in lib.letters]
        if missing:
            print(f"Letters with no image (shown as text): {' '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
