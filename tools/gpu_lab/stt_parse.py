"""Turn a raw speech transcript into the field the kiosk asked for (pure functions)."""
import re

_NAME_LEAD = re.compile(
    r"^\s*(?:hi|hello|hey)?[\s,]*(?:my\s+name\s+is|my\s+name's|name\s+is|this\s+is|it\s+is|it's|"
    r"i\s+am|i'm|im|call\s+me|myself|they\s+call\s+me|you\s+can\s+call\s+me)(?:\s+|$)", re.I)
_NAME_TAIL = re.compile(r"\s*(?:\b(?:sir|madam|please|thanks|thank\s+you)\b)\s*$", re.I)


def parse_name(text):
    t = _NAME_LEAD.sub("", text.strip())
    t = _NAME_TAIL.sub("", t)
    t = re.sub(r"[^A-Za-z .'\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .-'")
    if not t:
        return None
    def cap(w):                       # D'souza -> D'Souza, anne-marie -> Anne-Marie
        return re.sub(r"(^|[-'])([a-z])", lambda m: m.group(1) + m.group(2).upper(), w.lower())
    return " ".join(cap(w) for w in t.split(" "))[:60]


_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}


def parse_age(text):
    """Returns (age_in_years or None, months or None). '6 months' -> (0, 6)."""
    t = text.lower().replace("-", " ")
    m = re.search(r"\b(\d{1,3})\b", t)
    n = int(m.group(1)) if m else None
    if n is None:
        total, seen = 0, False
        for w in re.findall(r"[a-z]+", t):
            if w in _TENS:
                total += _TENS[w]; seen = True
            elif w in _ONES:
                total += _ONES[w]; seen = True
            elif w == "hundred" and seen:
                total = max(total, 1) * 100
        n = total if seen else None
    if n is None or n > 120:
        return None, None
    if re.search(r"\bmonths?\b", t) and not re.search(r"\byears?\b", t):
        return 0, n
    return n, None


def parse_sex(text):
    t = text.lower()
    if re.search(r"prefer not|rather not|don'?t want|skip|not say", t):
        return "unspecified"
    if re.search(r"\b(female|woman|girl|lady|she)\b", t):
        return "female"
    if re.search(r"\b(male|man|boy|gentleman|he)\b", t):
        return "male"
    if re.search(r"\b(other|non.?binary|trans)\b", t):
        return "other"
    return None


def parse_yesno(text):
    t = text.lower()
    if re.search(r"\b(no|nope|nah|wrong|incorrect|not right|not correct|change)\b", t):
        return "no"
    if re.search(r"\b(yes|yeah|yep|yup|correct|right|sure|ok|okay|that'?s it|exactly|affirmative)\b", t):
        return "yes"
    return None


PARSERS = {"name": parse_name, "age": parse_age, "sex": parse_sex, "yesno": parse_yesno}
