"""Question parsing, answer extraction, and response normalization."""
import re

# Same pattern as eval.py (copied, because importing eval.py would run it).
ANSWER_PATTERN = re.compile(r"\[ANSWER\]:\s*\(([A-J])\)")
ALL_LETTERS = list("ABCDEFGHIJ")
DEFAULT_LETTER = "A"


def option_letters(prompt):
    """Letters of the answer choices that appear in the prompt, e.g. ['A','B','C','D']."""
    # Prefer "(X)" at the start of a line, so in-text mentions such as a grade
    # "낙제(F)" are not mistaken for options; fall back to any "(X)".
    found = sorted(set(re.findall(r"(?:^|[\n ])[ \t]*\(([A-J])\)", prompt)))
    if not found:
        found = sorted(set(re.findall(r"\(([A-J])\)", prompt)))
    if not found:
        return ALL_LETTERS
    # Options are consecutive from A; fill gaps up to the largest letter seen.
    last = ALL_LETTERS.index(found[-1])
    return ALL_LETTERS[:last + 1]


def extract_letter(text, valid_letters):
    """Pull the chosen letter out of raw model output. Returns None if not found.

    Priority: last [ANSWER]: (X) tag -> 'answer is (X)' style -> last bare (X).
    """
    if not text:
        return None
    for pattern in (
        ANSWER_PATTERN,
        re.compile(r"answer\s*(?:is|:)?\s*\(?([A-J])\)", re.IGNORECASE),
        re.compile(r"\(([A-J])\)"),
    ):
        matches = [m for m in pattern.findall(text) if m in valid_letters]
        if matches:
            return matches[-1]
    return None


def format_response(letter):
    """Final CSV cell: exactly one tag, which eval.py can parse."""
    return f"[ANSWER]: ({letter})"


def has_hangul(text):
    return re.search(r"[가-힣]", text) is not None
