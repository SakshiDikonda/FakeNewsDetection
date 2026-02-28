import re
from typing import Optional

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<.*?>")
_MULTI_WS_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """
    Lightweight cleaner for news text.
    """
    if text is None:
        return ""
    text = str(text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text

def normalize_label(label) -> Optional[int]:
    """
    Map common label formats to 0/1:
    1 = REAL, 0 = FAKE
    """
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in {"real", "true", "1", "genuine", "legit"}:
        return 1
    if s in {"fake", "false", "0", "hoax"}:
        return 0
    return None
