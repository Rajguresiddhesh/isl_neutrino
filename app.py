"""SignConnect - two-way Indian Sign Language interpreter.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from signconnect import config
from signconnect.gloss_to_english import glosses_to_english
from signconnect.grammar import english_to_isl
from signconnect.landmarks import draw_hands
from signconnect.recognizer import UNKNOWN, SignRecognizer
from signconnect.speech import SpeechError, transcribe
from signconnect.text_to_sign import SignLibrary, render_sentences

st.set_page_config(page_title="SignConnect - ISL Interpreter", page_icon="🤟", layout="wide")

EXAMPLES = [
    "She is a good student.",
    "You are nice.",
    "Is she a good student?",
    "What is your name?",
    "Attention please. The train to Pune will arrive on platform 5.",
]


@st.cache_resource(show_spinner="Loading sign library...")
def get_library() -> SignLibrary:
    return SignLibrary()


@st.cache_resource(show_spinner="Loading recognition models...")
def get_recognizer() -> SignRecognizer:
    return SignRecognizer()


@st.cache_resource(show_spinner="Loading language model...")
def warm_up_grammar() -> bool:
    english_to_isl("Hello.")
    return True


# --------------------------------------------------------------------------- #
# Speech / text -> sign
# --------------------------------------------------------------------------- #
def show_translation(text: str) -> None:
    library = get_library()
    sentences = english_to_isl(text)
    if not sentences:
        st.warning("Nothing to translate - please enter a sentence.")
        return

    st.subheader("ISL sentence")
    for s in sentences:
        left, right = st.columns([1, 1])
        left.markdown(f"**English:** {s.source}")
        right.markdown(f"**ISL:** `{s.text}`")
        if s.facial_expression:
            right.caption(f"🙂 {s.facial_expression}")

    with st.spinner("Building the sign video..."):
        result = render_sentences(sentences, library=library)
    if result.path is None:
        st.error("No signs could be produced for this text.")
        return

    video_col, info_col = st.columns([3, 2])
    video_col.video(str(result.path), autoplay=True, loop=True, muted=True)
    info_col.metric("Signed from recorded clips", len(result.signed_from_video))
    info_col.metric("Fingerspelled (no clip yet)", len(result.fingerspelled))
    if result.fingerspelled:
        info_col.info("No sign clip exists yet for: **" + ", ".join(sorted(set(result.fingerspelled)))
                      + "**. These words were spelled letter by letter. Add a folder with that "
                      "word's clips to the dataset to sign them properly.")
    with open(result.path, "rb") as f:
        info_col.download_button("Download video", f.read(), file_name="isl_translation.mp4",
                                 mime="video/mp4")


def text_to_sign_tab() -> None:
    st.markdown("Type or speak English. It is re-ordered into ISL grammar and played as signs.")
    mode = st.radio("Input", ["Type text", "Record speech", "Upload audio"], horizontal=True,
                    label_visibility="collapsed")

    text = ""
    if mode == "Type text":
        choice = st.selectbox("Try an example", ["(write your own)"] + EXAMPLES)
        default = "" if choice.startswith("(") else choice
        text = st.text_area("English text", value=default, height=90,
                            placeholder="e.g. She is a good student.")
    else:
        if mode == "Record speech":
            audio = st.audio_input("Press the microphone and speak")
        else:
            audio = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
        if audio is not None:
            try:
                with st.spinner("Listening..."):
                    text = transcribe(audio.getvalue(), Path(audio.name or "a.wav").suffix or ".wav")
                st.success(f"Heard: “{text}”")
            except SpeechError as exc:
                st.error(str(exc))

    if st.button("Translate to ISL", type="primary", disabled=not text.strip()) or (
            mode != "Type text" and text.strip()):
        show_translation(text)


# --------------------------------------------------------------------------- #
# Sign -> text
# --------------------------------------------------------------------------- #
def video_to_text_section() -> None:
    recognizer = get_recognizer()
    st.markdown("Upload a video of someone signing. Each sign should end with the hands "
                "lowered or out of view.")
    st.caption("Signs this model knows: " + ", ".join(w.upper() for w in recognizer.vocabulary))
    upload = st.file_uploader("Signing video", type=[e.strip(".") for e in config.VIDEO_EXTENSIONS])
    as_question = st.checkbox("The signer used a questioning face (yes/no question)")
    if upload is None:
        return

    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as tmp:
        tmp.write(upload.getvalue())
        path = tmp.name
    try:
        with st.spinner("Tracking hands and recognising signs..."):
            result = recognizer.recognize_video(path)
    except Exception as exc:
        st.error(f"This video could not be processed: {exc}")
        return
    finally:
        Path(path).unlink(missing_ok=True)

    if not result.hands_seen:
        st.warning("No hands were detected. Make sure the signer's hands and face are visible "
                   "and well lit.")
        return
    if not result.predictions:
        st.warning("Hands were seen but never raised into the signing space.")
        return

    english = glosses_to_english(result.glosses, as_question)
    st.subheader("Translation")
    st.markdown(f"### {english or '—'}")
    st.markdown("**Signs seen:** " + " ".join(
        f"`{p.label.upper()}`" if p.label != UNKNOWN else "`?`" for p in result.predictions))
    st.dataframe(
        [{"From (s)": round(p.t_start, 1), "To (s)": round(p.t_end, 1),
          "Sign": p.label.upper() if p.label != UNKNOWN else "not recognised",
          "Confidence": f"{p.confidence:.0%}",
          "Closest known sign": p.best_guess.upper()} for p in result.predictions],
        hide_index=True, use_container_width=True)
    if not result.face_seen:
        st.caption("No face was found, so hand positions were measured from the frame centre; "
                   "accuracy is lower.")


def fingerspelling_section() -> None:
    recognizer = get_recognizer()
    st.markdown("Show one fingerspelled letter to the camera (or upload a photo) and add it "
                "to the word.")
    st.caption("Letters this model knows: " + " ".join(recognizer.letters))
    st.session_state.setdefault("spelled", "")

    source = st.radio("Picture from", ["Camera", "Upload"], horizontal=True,
                      label_visibility="collapsed")
    picture = (st.camera_input("Take a picture of the letter") if source == "Camera"
               else st.file_uploader("Letter image", type=["jpg", "jpeg", "png", "bmp", "webp"]))
    if picture is not None:
        image = cv2.imdecode(np.frombuffer(picture.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("That file is not a readable image.")
        else:
            letter, confidence, hands, ranked = recognizer.recognize_letter(image)
            left, right = st.columns([1, 1])
            if not hands:
                right.warning("No hand found in the picture.")
            else:
                right.metric("Letter", letter if letter != UNKNOWN else "unsure",
                             f"{confidence:.0%} confident")
                right.caption("Top guesses: " + ", ".join(f"{l} ({c:.0%})" for l, c in ranked))
                if right.button(f"Add “{ranked[0][0]}” to word"):
                    st.session_state.spelled += ranked[0][0]
            left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=280)

    st.markdown(f"**Word so far:** `{st.session_state.spelled or ' '}`")
    a, b, _ = st.columns([1, 1, 4])
    if a.button("⌫ Delete last") and st.session_state.spelled:
        st.session_state.spelled = st.session_state.spelled[:-1]
        st.rerun()
    if b.button("Clear"):
        st.session_state.spelled = ""
        st.rerun()


def sign_to_text_tab() -> None:
    mode = st.radio("Mode", ["Word signs (video)", "Fingerspelling (photo)"], horizontal=True,
                    label_visibility="collapsed")
    try:
        if mode.startswith("Word"):
            video_to_text_section()
        else:
            fingerspelling_section()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
    st.divider()
    st.caption("For continuous recognition from a webcam run  `python -m signconnect.live`  "
               "in a terminal.")


# --------------------------------------------------------------------------- #
def sidebar() -> None:
    library = get_library()
    st.sidebar.title("🤟 SignConnect")
    st.sidebar.caption("Two-way Indian Sign Language interpreter")
    st.sidebar.metric("Sign clips in dataset", len(library.clips))
    st.sidebar.write(", ".join(w.upper() for w in library.vocabulary) or "none found")
    st.sidebar.metric("Fingerspelling letters", len(library.letters))
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Add a new sign**\n\n"
        f"1. Create `{config.WORD_VIDEO_DIR.name}/<word>/` in the dataset folder\n"
        "2. Put clips of that sign inside (several signers and takes)\n"
        "3. Run `python -m signconnect.train_words`\n"
        "4. Restart the app")
    if st.sidebar.button("Rescan dataset"):
        library.refresh()
        st.rerun()


def main() -> None:
    sidebar()
    st.title("SignConnect")
    warm_up_grammar()
    to_sign, to_text = st.tabs(["🗣️ Speech / Text → Sign", "🤟 Sign → Text"])
    with to_sign:
        text_to_sign_tab()
    with to_text:
        sign_to_text_tab()


main()
