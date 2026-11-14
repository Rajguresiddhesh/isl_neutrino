# SignConnect — two-way Indian Sign Language interpreter

| Direction | Input | Output |
|---|---|---|
| **Speech / Text → Sign** | typed English, microphone, or an audio file | ISL-grammar gloss + a sign video |
| **Sign → Text** | a video of signing, a webcam, or a photo of a fingerspelled letter | English sentence |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Command line:

```bash
python -m signconnect text  "She is a good student."     # -> outputs/isl_*.mp4
python -m signconnect gloss "What does the student do?"  # -> STUDENT WHAT DO ?
python -m signconnect video gesture_folder/gesture1/newvideo.mov
python -m signconnect letter "ISL_Dataset 6.42.04 PM/B/B (1).jpg"
python -m signconnect live                               # webcam, continuous
python -m signconnect info
pytest                                                   # 43 tests
```

No paths need editing: everything is resolved relative to the repo
(`signconnect/config.py`), replacing the old hard-coded `/Users/...` paths.

## How it works

```
 SPEECH/TEXT -> SIGN
 audio --(speech.py: Google Web Speech)--> English text
       --(grammar.py: spaCy parse + ISL rules)--> ISL gloss   e.g. SHE STUDENT GOOD
       --(text_to_sign.py)--> clip per sign from  isl_sch 2/isl/<word>/
                              no clip? -> fingerspell with ISL_Dataset letter images
       --(OpenCV concat + H.264)--> video

 SIGN -> TEXT
 video/webcam --(landmarks.py: face box -> crop signer -> MediaPipe hands)--> 21 keypoints/hand
       --(segment.py: a sign = hands raised; rest = boundary)--> one chunk per sign
       --(recognizer.py: word model on the keypoint sequence)--> glosses
       --(gloss_to_english.py: SOV -> SVO, add is/are, articles, verb endings)--> English
```

### ISL grammar rules implemented (`signconnect/grammar.py`)

All 16 examples on the two AYJNISHD sheets are regression tests (`tests/test_grammar.py`).

| Rule | Example |
|---|---|
| Subject – Object – Verb | I go home → `I HOME GO` |
| Articles, is/am/are, helper verbs, "to" are dropped | He goes to school → `HE SCHOOL GO` |
| Adjective after its noun | She is a good student → `SHE STUDENT GOOD` |
| Possessive before the noun | Come to my house → `MY HOUSE COME` |
| Question word after subject/object, before the verb | What does the student do? → `STUDENT WHAT DO ?` |
| Yes/no question = statement order + questioning face | Is the food good? → `FOOD GOOD ?` |
| Time first, negation last | I did not eat today → `TODAY I EAT NOT` |
| Dictionary forms only (no tense/plural) | helps → `HELP`, students → `STUDENT` |

The last three rows go beyond the sheets and follow general ISL descriptions;
please have your ISL experts confirm them.

### Why keypoints instead of the earlier image CNN

The earlier scripts trained a CNN on whole 224×224 frames. With a handful of clips
that learns the signer's clothes and the room, not the sign. The recogniser now
uses MediaPipe hand keypoints (shape of each hand + where it is relative to the
face), which needs far less data, ignores background and runs on a laptop CPU.

## Adding vocabulary (the most important next step)

Text → Sign and Sign → Text both read the same folder, one sub-folder per word:

```
isl_sch 2/isl/
    teacher/  teacher_1.mp4  teacher_2.mp4 ...
    food/     ...
```

1. Add a folder named after the word with clips of that sign. Each clip: one sign,
   hands lowered at the start and end, face visible.
2. `python -m signconnect.train_words`
3. Restart the app.

Text → Sign needs one clip per word. Sign → Text needs **several different takes
by several signers** per word to be reliable.

## Retraining

```bash
python -m signconnect.train_alphabet   # letters, from ISL_Dataset 6.42.04 PM/
python -m signconnect.train_words      # word signs, from isl_sch 2/isl/
```

Run both if the bundled models fail to load (different scikit-learn version).

## Current limits — read before a demo

* **Vocabulary is 5 signs**: good, nice, she, student, you. Every other word is
  fingerspelled in the output video and cannot be recognised from video.
* **Each word has one real recording.** The 5 files in each word folder are
  byte-identical copies, and `newvideo.mov` is made from the same footage. The
  word model therefore cannot be given an honest accuracy number yet; the trainer
  prints leave-one-out accuracy automatically once every word has ≥ 2 distinct clips.
* **Letters:** 23 of 26 (no H, J, Y images in the dataset). Cross-validated accuracy
  is 76% on images where a hand was found (67% counting images with no hand found),
  and the images are hand cut-outs on artificial backgrounds, so expect lower on a
  real camera. More real photos per letter is the fix.
* **Facial expression is not read.** A yes/no question looks like a statement to
  the camera; the app has a checkbox for it.
* **Gloss → English** covers the simple sentence patterns on the sheets; other
  sign sequences are shown as plain glosses rather than guessed.
* **Speech input** needs internet (Google Web Speech) and English.
* `python -m signconnect.live` was tested on video files, not on a physical webcam.

## Repository layout

```
app.py                      Streamlit app (both directions)
signconnect/
  config.py                 all paths and thresholds
  grammar.py                English -> ISL gloss
  gloss_to_english.py       ISL gloss -> English
  text_to_sign.py           gloss -> sign video
  speech.py                 audio -> text
  landmarks.py              hand tracking + features
  segment.py                splits a video into signs
  recognizer.py             word and letter recognition
  train_words.py            trains models/word_model.joblib
  train_alphabet.py         trains models/alphabet_model.joblib
  live.py                   webcam mode
  __main__.py               command line
models/                     trained models + MediaPipe / OpenCV detector files
tests/                      pytest suite
```

The older scripts (`completeprog.py`, `nlp_text_to_vid.py`, `video_to_text/`) are
superseded by the package above and can be deleted. `text_to_isl_model.pkl` (62 MB)
is not used.
