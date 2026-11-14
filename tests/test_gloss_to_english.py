"""Round trip: English -> ISL gloss -> English for the grammar-sheet examples."""
import pytest

from signconnect.gloss_to_english import glosses_to_english
from signconnect.grammar import english_to_isl

# (sheet sentence, sentence expected back).  Plurals cannot survive the trip
# because ISL glosses do not mark them.
ROUND_TRIPS = [
    ("The teacher helps students.", "The teacher helps a student."),
    ("The student reads a book.", "The student reads a book."),
    ("I go home.", "I go home."),
    ("The food tastes good.", "The food tastes good."),
    ("They eat food.", "They eat food."),
    ("What is your name?", "What is your name?"),
    ("She is a good student.", "She is a good student."),
    ("The food tastes bad.", "The food tastes bad."),
    ("He goes to school.", "He goes to school."),
    ("Come to my house.", "Come to my house."),
    ("Is the teacher nice?", "Is the teacher nice?"),
    ("What does the student do?", "What does the student do?"),
    ("Where is your home?", "Where is your home?"),
    ("Is the food good?", "Is the food good?"),
    ("Do you eat food?", "Do you eat food?"),
    ("You are nice.", "You are nice."),
    ("She does not eat food.", "She does not eat food."),
]


@pytest.mark.parametrize("english,expected", ROUND_TRIPS)
def test_round_trip(english, expected):
    isl = english_to_isl(english)[0]
    assert glosses_to_english(isl.glosses, isl.question == "yes_no") == expected


def test_unknown_pattern_falls_back_to_glosses():
    assert glosses_to_english(["student", "teacher", "good", "nice"]) == "Student teacher good nice."


def test_empty():
    assert glosses_to_english([]) == ""
