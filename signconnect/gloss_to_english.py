"""ISL gloss sequence -> readable English sentence (the reverse of grammar.py).

The recogniser outputs signs in ISL order, e.g. SHE STUDENT GOOD.  This module
puts back what ISL leaves out (word order, "is/are", articles, verb endings)
for the simple sentence patterns on the grammar sheets.  Anything it does not
recognise is returned as the plain gloss sequence rather than guessed at.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
OBJECT_FORM = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}
POSSESSIVES = {"my", "your", "his", "her", "our", "their", "its"}
WH_WORDS = {"what", "where", "who", "when", "why", "how", "which"}
ADJECTIVES = {
    "good", "bad", "nice", "big", "small", "happy", "sad", "hot", "cold", "new", "old",
    "beautiful", "tall", "short", "late", "early", "hungry", "thirsty", "sick", "tired",
    "fine", "clean", "dirty", "fast", "slow", "easy", "difficult", "tasty", "angry", "busy",
    "red", "blue", "green", "black", "white", "yellow", "deaf", "ready", "right", "wrong",
}
VERBS = {
    "go", "come", "eat", "drink", "read", "write", "help", "taste", "do", "see", "look",
    "like", "love", "want", "need", "have", "give", "take", "play", "work", "study", "learn",
    "teach", "sleep", "sit", "stand", "walk", "run", "open", "close", "buy", "sell", "cook",
    "wash", "call", "meet", "know", "understand", "live", "wait", "stop", "start", "arrive",
    "leave", "speak", "sign", "listen", "watch", "make", "bring", "ask", "tell", "pay",
}
MOTION_VERBS = {"go", "come", "walk", "run", "arrive"}
PEOPLE = {
    "teacher", "student", "man", "woman", "boy", "girl", "mother", "father", "brother",
    "sister", "friend", "doctor", "child", "baby", "people", "person", "family", "driver",
}
MASS_NOUNS = {"food", "water", "milk", "rice", "tea", "coffee", "money", "work", "music",
              "homework", "bread", "sugar", "salt", "time", "help"}
NO_PREPOSITION = {"home", "here", "there"}
BARE_PLACES = {"school", "college", "work", "church", "bed", "hospital", "market"}
IRREGULAR_3SG = {"have": "has", "do": "does", "go": "goes"}


def _tag(word: str) -> str:
    if word in SUBJECT_PRONOUNS:
        return "PRON"
    if word in POSSESSIVES:
        return "POSS"
    if word in WH_WORDS:
        return "WH"
    if word == "not":
        return "NEG"
    if word in ADJECTIVES:
        return "ADJ"
    if word in VERBS:
        return "VERB"
    return "NOUN"


def _third_person(verb: str) -> str:
    if verb in IRREGULAR_3SG:
        return IRREGULAR_3SG[verb]
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def _be(subject: Sequence[str]) -> str:
    if subject == ["i"]:
        return "am"
    if len(subject) == 1 and subject[0] in ("you", "we", "they"):
        return "are"
    return "is"


def _is_third_singular(subject: Sequence[str]) -> bool:
    return not (len(subject) == 1 and subject[0] in ("i", "you", "we", "they"))


def _subject_text(subject: Sequence[str]) -> str:
    if len(subject) == 1 and subject[0] in SUBJECT_PRONOUNS:
        return "I" if subject[0] == "i" else subject[0]
    if subject[0] in POSSESSIVES:
        return " ".join(subject)
    return "the " + " ".join(subject)


def _object_text(obj: Sequence[str]) -> str:
    if len(obj) == 1 and obj[0] in SUBJECT_PRONOUNS:
        return OBJECT_FORM.get(obj[0], obj[0])
    if obj[0] in POSSESSIVES or obj[-1] in MASS_NOUNS:
        return " ".join(obj)
    return _article(obj[0]) + " " + " ".join(obj)


def _finish(text: str, question: bool) -> str:
    text = " ".join(text.split())
    return text[0].upper() + text[1:] + ("?" if question else ".")


def _take_noun_phrase(words: List[str], tags: List[str], i: int):
    """Read [possessive] noun | pronoun starting at i. Returns (phrase, next_i)."""
    if i < len(words) and tags[i] == "PRON":
        return [words[i]], i + 1
    j = i
    if j < len(words) and tags[j] == "POSS":
        j += 1
    if j < len(words) and tags[j] == "NOUN":
        return words[i:j + 1], j + 1
    return None, i


def glosses_to_english(glosses: Sequence[str], yes_no_question: bool = False) -> str:
    """Turn recognised signs into an English sentence.

    `yes_no_question` should be True when the signer used a questioning facial
    expression; the camera pipeline cannot see that yet, so callers decide.
    """
    words = [g.lower() for g in glosses if g and g != "?"]
    if not words:
        return ""
    tags = [_tag(w) for w in words]
    fallback = _finish(" ".join(words), False)

    negated = "NEG" in tags
    if negated:
        keep = [i for i, t in enumerate(tags) if t != "NEG"]
        words, tags = [words[i] for i in keep], [tags[i] for i in keep]
        if not words:
            return "No."

    subject, i = _take_noun_phrase(words, tags, 0)
    rest, rest_tags = words[i:], tags[i:]

    # -- imperative: "MY HOUSE COME" -> "Come to my house." --------------------
    if (subject and rest_tags == ["VERB"] and rest[0] in MOTION_VERBS
            and tags[0] != "PRON" and subject[-1] not in PEOPLE):
        place = " ".join(subject)
        link = "" if subject[-1] in NO_PREPOSITION else "to "
        prefix = "do not " if negated else ""
        return _finish(f"{prefix}{rest[0]} {link}{place}", False)
    if subject is None:
        if tags == ["VERB"]:
            return _finish(("do not " if negated else "") + words[0], False)
        return fallback

    subj = _subject_text(subject)
    be = _be(subject)
    third = _is_third_singular(subject)
    do = "does" if third else "do"
    not_ = " not" if negated else ""

    def verb_phrase(verb: str) -> str:
        if negated:
            return f"{do} not {verb}"
        return _third_person(verb) if third else verb

    # -- questions with a question word ---------------------------------------
    if "WH" in rest_tags:
        k = rest_tags.index("WH")
        wh, after = rest[k], rest[k + 1:]
        before = rest[:k]
        if not after and not before:                       # YOUR NAME WHAT
            return _finish(f"{wh} {be} {subj}", True)
        if not after and rest_tags[:k] == ["ADJ"]:
            return _finish(f"{wh} {be} {subj} {before[0]}", True)
        if len(after) == 1 and rest_tags[k + 1] == "VERB":  # STUDENT WHAT DO
            obj = (" " + _object_text(before)) if before and rest_tags[0] in ("NOUN", "POSS", "PRON") else ""
            return _finish(f"{wh} {do} {subj} {after[0]}{obj}", True)
        return _finish(" ".join(words), True)

    # -- no verb: "is/are" sentences ------------------------------------------
    if "VERB" not in rest_tags:
        if rest_tags == ["ADJ"]:                            # FOOD GOOD
            body = f"{subj} {be}{not_} {rest[0]}"
            if yes_no_question:
                body = f"{be} {subj}{not_} {rest[0]}"
        elif sorted(rest_tags) == ["ADJ", "NOUN"]:          # SHE STUDENT GOOD (or GOOD STUDENT)
            adj, head = (rest[1], rest[0]) if rest_tags[0] == "NOUN" else (rest[0], rest[1])
            noun = f"{_article(adj)} {adj} {head}"
            body = f"{be} {subj}{not_} {noun}" if yes_no_question else f"{subj} {be}{not_} {noun}"
        elif rest_tags == ["NOUN"] or rest_tags == ["POSS", "NOUN"]:
            noun = _object_text(rest)
            body = f"{be} {subj}{not_} {noun}" if yes_no_question else f"{subj} {be}{not_} {noun}"
        elif not rest:
            return fallback
        else:
            return fallback
        return _finish(body, yes_no_question)

    # -- verb-final sentences -------------------------------------------------
    if rest_tags[-1] != "VERB":
        return fallback
    verb, middle, middle_tags = rest[-1], rest[:-1], rest_tags[:-1]
    if not middle:                                          # STUDENT READ
        tail = ""
    elif middle_tags == ["ADJ"]:                            # FOOD GOOD TASTE
        tail = " " + middle[0]
    elif middle_tags in (["NOUN"], ["POSS", "NOUN"], ["PRON"]):
        if verb in MOTION_VERBS:                            # HE SCHOOL GO / I HOME GO
            if middle[-1] in NO_PREPOSITION:
                tail = " " + " ".join(middle)
            elif middle[0] in POSSESSIVES or middle[-1] in BARE_PLACES:
                tail = " to " + " ".join(middle)
            else:
                tail = " to the " + " ".join(middle)
        else:                                               # STUDENT BOOK READ
            tail = " " + _object_text(middle)
    elif middle_tags == ["NOUN", "ADJ"]:                    # I BOOK NEW WANT
        tail = f" {_article(middle[1])} {middle[1]} {middle[0]}"
    else:
        return fallback
    if yes_no_question:
        return _finish(f"{do} {subj}{not_} {verb}{tail}", True)
    return _finish(f"{subj} {verb_phrase(verb)}{tail}", False)
