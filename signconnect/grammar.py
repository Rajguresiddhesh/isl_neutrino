"""English -> ISL gloss conversion.

Implements the grammar rules from the AYJNISHD reference sheets:

  * word order is Subject - Object - Verb          "I go home"            -> I HOME GO
  * articles, "is/am/are", helper verbs and "to"
    are not signed                                 "He goes to school"    -> HE SCHOOL GO
  * an adjective follows its noun                  "She is a good student"-> SHE STUDENT GOOD
  * possessives stay in front of the noun          "Come to my house"     -> MY HOUSE COME
  * a question word comes after the subject/object,
    just before the verb if there is one           "What is your name?"   -> YOUR NAME WHAT
                                                   "What does the student do?" -> STUDENT WHAT DO
  * yes/no questions keep statement order and are
    marked only by a questioning facial expression "Is the food good?"    -> FOOD GOOD ?
  * time words lead the sentence, negation ends it "I did not eat today"  -> TODAY I EAT NOT

Signs carry no tense, plural or person endings, so every word is reduced to its
dictionary form ("helps" -> HELP, "students" -> STUDENT).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

DROPPED_PREPOSITIONS = {"to", "of", "at", "by", "into", "onto", "upon"}
DROPPED_DETERMINERS = {"a", "an", "the"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
MODAL_GLOSS = {"could": "can", "might": "may", "would": "will", "shall": "will"}
NEGATIONS = {"not", "n't", "never", "no"}
TIME_WORDS = {
    "today", "tomorrow", "yesterday", "now", "tonight", "later", "soon", "morning",
    "afternoon", "evening", "night", "daily", "always", "sometimes", "before", "after",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
WH_TAGS = {"WP", "WRB", "WDT", "WP$"}
CLAUSE_DEPS = {"ROOT", "conj", "advcl", "ccomp", "parataxis"}
# Greetings and set phrases are signed as units, not re-ordered.
FIXED_PHRASES = {
    "thank you": ["thank", "you"],
    "thank you very much": ["thank", "you", "very", "much"],
    "thanks": ["thank", "you"],
    "good morning": ["good", "morning"],
    "good afternoon": ["good", "afternoon"],
    "good evening": ["good", "evening"],
    "good night": ["good", "night"],
    "how are you": ["you", "how"],
}
PRONOUN_GLOSS = {"me": "i", "him": "he", "them": "they", "us": "we",
                 "'m": "", "am": "", "mine": "my", "yours": "your"}


@dataclass
class ISLSentence:
    source: str                       # the English sentence
    glosses: List[str]                # signs, in signing order (lower-case dictionary forms)
    question: Optional[str] = None    # None, "yes_no" or "wh"
    notes: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Conventional gloss notation: capitals, '?' for questions."""
        out = " ".join(g.upper() for g in self.glosses)
        return out + " ?" if self.question and out else out

    @property
    def facial_expression(self) -> Optional[str]:
        if self.question == "yes_no":
            return "Questioning face: eyebrows raised, head tilted forward"
        if self.question == "wh":
            return "Questioning face: eyebrows furrowed on the question sign"
        return None


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy once. Uses the medium model if installed, otherwise the small one."""
    import spacy

    for name in ("en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(name)
        except OSError:
            continue
    raise RuntimeError(
        "No spaCy English model is installed. Run:  python -m spacy download en_core_web_sm")


# --------------------------------------------------------------------------- #
def _is_time(tok) -> bool:
    return (tok.lemma_.lower() in TIME_WORDS and tok.pos_ in ("NOUN", "ADV", "PROPN")
            ) or (tok.ent_type_ in ("DATE", "TIME") and tok.dep_ in ("npadvmod", "advmod"))


def _is_wh(tok) -> bool:
    return tok.tag_ in WH_TAGS and tok.dep_ != "relcl" and tok.lemma_.lower() != "that"


def _is_signed(tok) -> bool:
    """Does this English word correspond to a sign at all?"""
    low = tok.text.lower()
    if tok.is_punct or tok.is_space or tok.pos_ in ("PUNCT", "SYM", "SPACE", "X"):
        return bool(re.search(r"\d", tok.text))
    if tok.pos_ == "DET" and low in DROPPED_DETERMINERS:
        return False
    if tok.pos_ == "AUX":                      # is/are/was, do/does/did, have/has (helpers)
        return low in MODALS or tok.lemma_.lower() in MODALS
    if tok.pos_ == "PART":                     # infinitive "to", possessive "'s"
        return low in NEGATIONS
    if tok.pos_ == "ADP" and low in DROPPED_PREPOSITIONS:
        return False
    if tok.pos_ == "CCONJ" and low == "and":
        return False
    if tok.pos_ == "SCONJ" and low in ("that", "whether"):
        return False
    if tok.lemma_.lower() == "be":             # copula, however it was tagged
        return False
    if tok.dep_ == "expl":                     # "there is ..."
        return False
    return True


def _gloss(tok) -> str:
    low = tok.text.lower()
    if low in PRONOUN_GLOSS:
        return PRONOUN_GLOSS[low]
    if tok.pos_ == "PRON" or tok.tag_ in ("PRP", "PRP$"):
        return low                             # keep I / my / your / she distinct
    if low in ("n't", "not"):
        return "not"
    if tok.like_num or tok.pos_ == "NUM":
        return low
    if tok.pos_ == "PROPN":
        return low                             # names are fingerspelled as written
    lemma = tok.lemma_.lower().strip()
    lemma = MODAL_GLOSS.get(lemma, lemma)
    return re.sub(r"[^a-z0-9'-]", "", lemma) or low


def _clause_head(tok):
    """The verb (or copula) that governs the clause this token belongs to."""
    cur = tok
    while True:
        if cur.dep_ in CLAUSE_DEPS and cur.pos_ in ("VERB", "AUX"):
            return cur
        if cur.head.i == cur.i:
            return cur
        cur = cur.head


def _np_order_key(tok):
    """Sort key that moves adjectives and numbers to just after their noun."""
    if tok.dep_ in ("amod", "nummod") and tok.head.pos_ in ("NOUN", "PROPN") and tok.i < tok.head.i:
        return (tok.head.i, 1 if tok.dep_ == "amod" else 2, tok.i)
    return (tok.i, 0, tok.i)


def _convert_clause(head, tokens) -> List[str]:
    # NB: spaCy creates a new Token object on every access, so compare by index.
    subject = next((t for t in tokens
                    if t.dep_ in ("nsubj", "nsubjpass") and t.head.i == head.i), None)
    subject_span = {t.i for t in subject.subtree} if subject is not None else set()

    lead, time, subj, rest, wh, verb, modal, neg = [], [], [], [], [], [], [], []
    for tok in tokens:
        if not _is_signed(tok):
            continue
        low = tok.text.lower()
        if _is_wh(tok):
            wh.append(tok)
        elif tok.dep_ == "mark" or (tok.dep_ == "cc" and tok.pos_ == "CCONJ"):
            lead.append(tok)                   # because / if / but open their clause
        elif low in NEGATIONS and tok.dep_ in ("neg", "det", "advmod"):
            neg.append(tok)
        elif tok.pos_ == "AUX":
            modal.append(tok)
        elif tok.i == head.i and tok.pos_ == "VERB":
            verb.append(tok)
        elif tok.dep_ == "prt" and tok.head.i == head.i:
            verb.append(tok)
        elif _is_time(tok):
            time.append(tok)
        elif tok.i in subject_span:
            subj.append(tok)
        else:
            rest.append(tok)

    ordered = (lead + time + sorted(subj, key=_np_order_key) + sorted(rest, key=_np_order_key)
               + wh + sorted(verb, key=lambda t: t.i) + modal + neg)
    return [g for g in (_gloss(t) for t in ordered) if g]


def _convert_sentence(sent) -> ISLSentence:
    tokens = [t for t in sent]
    plain = re.sub(r"[^a-z ]", "", sent.text.lower()).strip()
    if plain in FIXED_PHRASES:
        question = "wh" if sent.text.strip().endswith("?") else None
        return ISLSentence(sent.text.strip(), list(FIXED_PHRASES[plain]), question)
    clauses = {}
    for tok in tokens:
        clauses.setdefault(_clause_head(tok).i, []).append(tok)

    glosses: List[str] = []
    for head_i in sorted(clauses, key=lambda i: min(t.i for t in clauses[i])):
        glosses.extend(_convert_clause(sent.doc[head_i], clauses[head_i]))

    question = None
    if any(_is_wh(t) for t in tokens) and (sent.text.strip().endswith("?")
                                           or tokens[0].tag_ in WH_TAGS):
        question = "wh"
    elif sent.text.strip().endswith("?") or (
            tokens[0].pos_ == "AUX" and any(t.dep_ == "nsubj" and t.i > tokens[0].i for t in tokens)):
        question = "yes_no"

    notes = []
    if question == "yes_no":
        notes.append("Yes/no question: same sign order as a statement, shown with a "
                     "questioning facial expression.")
    return ISLSentence(sent.text.strip(), glosses, question, notes)


def english_to_isl(text: str) -> List[ISLSentence]:
    """Convert English text (one or more sentences) to ISL gloss sentences."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    doc = get_nlp()(text)
    return [s for s in (_convert_sentence(sent) for sent in doc.sents) if s.glosses]
