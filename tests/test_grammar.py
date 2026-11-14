"""Every example on the two AYJNISHD grammar sheets, as a regression test.

The sheets write inflected English words ("Teacher students helps"); ISL signs
have no inflection, so the engine outputs dictionary forms (TEACHER STUDENT HELP).
"""
import pytest

from signconnect.grammar import english_to_isl

SHEET_SENTENCES = [
    ("The teacher helps students.", "teacher student help", None),
    ("The student reads a book.", "student book read", None),
    ("I go home.", "i home go", None),
    ("The food tastes good.", "food good taste", None),
    ("They eat food.", "they food eat", None),
    ("What is your name?", "your name what", "wh"),
    ("She is a good student.", "she student good", None),
    ("The food tastes bad.", "food bad taste", None),
    ("He goes to school.", "he school go", None),
    ("Come to my house.", "my house come", None),
]
SHEET_QUESTIONS = [
    ("Is the teacher nice?", "teacher nice", "yes_no"),
    ("What does the student do?", "student what do", "wh"),
    ("Where is your home?", "your home where", "wh"),
    ("Is the food good?", "food good", "yes_no"),
    ("Do you eat food?", "you food eat", "yes_no"),
    ("What is your name?", "your name what", "wh"),
]
EXTRA = [
    ("I did not eat today.", "today i eat not", None),
    ("You are nice.", "you nice", None),
    ("Is she a good student?", "she student good", "yes_no"),
    ("The train will arrive at platform 5.", "train platform 5 arrive will", None),
]


@pytest.mark.parametrize("english,expected,question", SHEET_SENTENCES + SHEET_QUESTIONS + EXTRA)
def test_sheet_example(english, expected, question):
    result = english_to_isl(english)
    assert len(result) == 1
    assert " ".join(result[0].glosses) == expected
    assert result[0].question == question


def test_multiple_sentences():
    result = english_to_isl("She is a good student. What is your name?")
    assert [s.text for s in result] == ["SHE STUDENT GOOD", "YOUR NAME WHAT ?"]


def test_empty():
    assert english_to_isl("   ") == []
