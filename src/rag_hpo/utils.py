import re

_PARENS_PATTERN = re.compile(r"\s*\([^)]*\)\s*")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TRAILING_PUNCT_PATTERN = re.compile(r"[^\w\s]+$")
_NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]+")
_WORD_PATTERN = re.compile(r"\w+")
_JSON_OBJECT_PATTERN = re.compile(r"\{.*}", re.DOTALL)


def _clean_text(txt: str) -> str:
    txt = _PARENS_PATTERN.sub(" ", txt)
    txt = _WHITESPACE_PATTERN.sub(" ", txt).strip().lower()
    return _TRAILING_PUNCT_PATTERN.sub("", txt)
