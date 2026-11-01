"""Turn spoken Vietnamese numbers into digits in an ASR transcript.

Whisper writes what it hears, so "muộn sáu mươi phút"; the labels and every numeric tool argument
use digits, so "muộn 60 phút". Measured on the labelled `train` split, a meaningful part of the
apparent ASR word-error rate is this difference and not a mishearing at all — see EXPERIMENTS.md.

It matters beyond the WER figure. A numeric tool argument has to come out exactly right
(`{"amount": 50000}`), and a string argument matched with `x-match: "fuzzy"` is scored by token-F1
against gold text that is written with digits, so words where gold has digits lose overlap. Doing
the conversion here, deterministically, is one less thing for the LLM to get wrong.

Deliberately conservative: a run of number words is rewritten only when it parses cleanly, and
anything ambiguous is left exactly as it was. A wrong number is worse than a spelled-out one.
"""
from __future__ import annotations

import re
import unicodedata

UNITS = {"một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
         "năm": 5, "lăm": 5, "nhăm": 5, "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
# Digit-by-digit reading, used for phone numbers and IDs: "không chín một chín" is 0919, NOT
# 0+9+1+9. Getting this wrong corrupts a tool argument and loses the sample outright, so a run of
# three or more bare digit words is concatenated instead of added up.
DIGITS = {"không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4", "năm": "5",
          "lăm": "5", "sáu": "6", "bảy": "7", "bẩy": "7", "tám": "8", "chín": "9"}
DIGIT_RUN_MINIMUM = 3
# Vietnamese names people by birth order: "Bác Ba", "Cô Tư", "Anh Hai" are persons, not numbers.
# Rewriting "bác ba" to "bác 3" breaks the gold argument it is meant to help: a fuzzy-matched
# query scores token-F1 {bác, 3} against {bác, ba} = 0.5, below the 0.6 threshold, so the call
# stops matching. After one of these words, a lone digit word stays a word.
TITLES = {"bác", "chú", "cô", "dì", "cậu", "mợ", "thím", "anh", "chị", "em", "ông", "bà",
          "thầy", "sếp", "chef", "mr", "ms",
          # Familiar forms are just as common in speech: "thằng Hai", "con Ba", "cháu Tư".
          # Found by inspecting a real transcript that this rewrote to "đèn thằng 2".
          "thằng", "con", "cháu", "bé", "chàng", "nàng", "lão", "mẹ", "bố", "ba", "má"}
# "không" is deliberately absent from UNITS: on its own it is the negation "not", not the number
# zero. The train labels normalise it to "0" anyway ("0 mưa thì thôi" for "không mưa thì thôi"),
# which flatters WER while making the text harder for the LLM to read. Chase the score, not the
# label's quirk.
SCALES = {"nghìn": 1000, "ngàn": 1000, "triệu": 10 ** 6, "tỷ": 10 ** 9, "tỉ": 10 ** 9}
ARITHMETIC_ONLY = {"mười", "mươi", "trăm", "rưỡi", "linh", "lẻ"}
TENS = "mươi"
TEN = "mười"
HUNDRED = "trăm"
ZEROES = {"linh", "lẻ"}
# "rưỡi" means half of the preceding scale: "hai triệu rưỡi" = 2_500_000.
HALF = "rưỡi"
NUMBER_WORDS = set(UNITS) | set(DIGITS) | set(SCALES) | {TEN, TENS, HUNDRED, HALF} | ZEROES


def _parse(tokens: list[str], start: int) -> tuple[int | None, int]:
    """Read one number starting at `tokens[start]`. Returns (value, index after it)."""
    total = section = 0
    index = start
    seen = False
    while index < len(tokens):
        token = tokens[index]
        if token in UNITS:
            value = UNITS[token]
            index += 1
            seen = True
            if index < len(tokens) and tokens[index] == TENS:
                section += value * 10
                index += 1
                if index < len(tokens) and tokens[index] in UNITS and tokens[index] not in SCALES:
                    section += UNITS[tokens[index]]
                    index += 1
            elif index < len(tokens) and tokens[index] == HUNDRED:
                section += value * 100
                index += 1
                if index < len(tokens) and tokens[index] in ZEROES:
                    index += 1
                    if index < len(tokens) and tokens[index] in UNITS:
                        section += UNITS[tokens[index]]
                        index += 1
            else:
                section += value
            continue
        if token == TEN:
            section += 10
            index += 1
            seen = True
            if index < len(tokens) and tokens[index] in UNITS:
                section += UNITS[tokens[index]]
                index += 1
            continue
        if token in SCALES:
            if not seen and section == 0:
                return None, start      # a bare "nghìn" is not a number
            total += (section or 1) * SCALES[token]
            section = 0
            index += 1
            seen = True
            continue
        if token == HALF:
            # Half of the last scale used, so only meaningful right after one.
            if total and not section:
                scale = max(s for s in SCALES.values() if total % s == 0) if total else 0
                if scale:
                    total += scale // 2
                    index += 1
                    continue
            return (total + section, index) if seen else (None, start)
        break
    if not seen:
        return None, start
    return total + section, index


def spoken_numbers_to_digits(text: str) -> str:
    """Rewrite every cleanly parsable run of Vietnamese number words as digits."""
    if not text:
        return text
    pieces = re.split(r"(\W+)", unicodedata.normalize("NFC", text))
    tokens = [p.lower() for p in pieces]
    out: list[str] = []
    index = 0
    while index < len(pieces):
        if tokens[index] in NUMBER_WORDS:
            # Collect this run of number words, keeping the separators that sit between them.
            words, positions = [], []
            scan = index
            while scan < len(pieces):
                if tokens[scan] in NUMBER_WORDS:
                    words.append(tokens[scan])
                    positions.append(scan)
                    scan += 1
                elif re.fullmatch(r"[ \t]+", pieces[scan]) and scan + 1 < len(pieces) \
                        and tokens[scan + 1] in NUMBER_WORDS:
                    scan += 1
                else:
                    break
            end = positions[-1] + 1
            # "Bác Ba" is a person, so a lone digit word after a kinship term or title stays a
            # word -- unless it opens a larger number, where "bác ba mươi tuổi" really is 30.
            previous = next((tokens[j] for j in range(index - 1, -1, -1) if pieces[j].strip()), "")
            builds_bigger = len(words) > 1 and (words[1] in {TENS, HUNDRED} or words[1] in SCALES)
            if previous in TITLES and words[0] in DIGITS and not builds_bigger:
                out.append(pieces[index])
                index += 1          # resume at the next word: the rest of the run may be a number
                continue
            # A long run of bare digit words is a phone number or an ID, read one digit at a time.
            if len(words) >= DIGIT_RUN_MINIMUM and all(w in DIGITS for w in words):
                out.append("".join(DIGITS[w] for w in words))
                index = end
                continue
            # Two bare digit words are genuinely ambiguous: "hai ba" is 23 or the digits 2 3, and
            # never 5. Adding them up is the one reading that is certainly wrong, so refuse.
            ambiguous = len(words) > 1 and all(w in DIGITS for w in words)
            value, used = _parse(words, 0)
            # Require the run to parse as ONE number; a leftover tail means the reading was unclear.
            if not ambiguous and value is not None and used == len(words):
                out.append(str(value))
                index = end
                continue
            # Refused. Emit the whole run untouched and move past it: re-entering it word by word
            # would convert its tail ("hai ba" -> "hai 3"), which is the error we just rejected.
            out.extend(pieces[index:end])
            index = end
            continue
        out.append(pieces[index])
        index += 1
    return "".join(out)
