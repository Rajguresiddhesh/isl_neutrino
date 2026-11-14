"""Speech -> text for the audio input path (announcements, spoken sentences).

Uses the free Google Web Speech API through the SpeechRecognition package, so
it needs an internet connection.  Any audio format ffmpeg can read is accepted.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path


class SpeechError(RuntimeError):
    pass


def to_wav(audio_bytes: bytes, suffix: str = ".wav") -> bytes:
    """Convert any audio (mp3, m4a, webm, wav...) to 16 kHz mono PCM WAV."""
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix or '.bin'}"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(audio_bytes)
        done = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)],
            capture_output=True, text=True)
        if done.returncode != 0 or not dst.exists():
            raise SpeechError(f"Could not read the audio file: {done.stderr.strip()[:200]}")
        return dst.read_bytes()


def transcribe(audio_bytes: bytes, suffix: str = ".wav", language: str = "en-IN") -> str:
    """Return the spoken words in `audio_bytes` as text."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise SpeechError("Speech input needs:  pip install SpeechRecognition") from exc

    wav = to_wav(audio_bytes, suffix)
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav)) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError as exc:
        raise SpeechError("No speech could be understood in the recording.") from exc
    except sr.RequestError as exc:
        raise SpeechError(f"The speech service could not be reached ({exc}). "
                          "Check the internet connection.") from exc
