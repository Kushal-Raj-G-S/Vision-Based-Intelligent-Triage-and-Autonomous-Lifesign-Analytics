"""Unit tests for the transcript parsers. Run: python tools/gpu_lab/test_stt_parse.py"""
from stt_parse import parse_name, parse_age, parse_sex, parse_yesno

fails = 0


def eq(label, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(("PASS " if ok else "FAIL ") + f"{label}: {got!r}" + ("" if ok else f"  (wanted {want!r})"))


# names
eq("plain", parse_name("Manoj Kumar"), "Manoj Kumar")
eq("my name is", parse_name("My name is Priya Sharma."), "Priya Sharma")
eq("I'm ... please", parse_name("I'm Aarav Mehta, please"), "Aarav Mehta")
eq("hello this is", parse_name("Hello, this is lakshmi narayanan"), "Lakshmi Narayanan")
eq("initials+apostrophe", parse_name("it's D'Souza"), "D'Souza")
eq("hyphen", parse_name("Anne-Marie"), "Anne-Marie")
eq("empty", parse_name("   "), None)
eq("filler only", parse_name("my name is"), None)
# ages
eq("digits", parse_age("I am 34 years old"), (34, None))
eq("words", parse_age("thirty four"), (34, None))
eq("hyphen words", parse_age("sixty-one"), (61, None))
eq("teen", parse_age("seventeen"), (17, None))
eq("months", parse_age("six months"), (0, 6))
eq("nonsense", parse_age("banana"), (None, None))
eq("too old", parse_age("500"), (None, None))
# sex
eq("male", parse_sex("I'm male"), "male")
eq("female", parse_sex("female"), "female")
eq("woman", parse_sex("a woman"), "female")
eq("prefer not", parse_sex("I'd prefer not to say"), "unspecified")
eq("female beats male substring", parse_sex("female"), "female")
eq("unclear", parse_sex("hmm"), None)
# yes/no
eq("yes", parse_yesno("Yes, that's right"), "yes")
eq("no", parse_yesno("No that's wrong"), "no")
eq("no beats right", parse_yesno("no, that is not right"), "no")
eq("okay", parse_yesno("okay"), "yes")
eq("unclear", parse_yesno("maybe"), None)
print("ALL PASS" if not fails else f"{fails} FAILED")
raise SystemExit(1 if fails else 0)
