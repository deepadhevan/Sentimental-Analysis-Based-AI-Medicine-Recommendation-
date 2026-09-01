import re

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_aspect(aspect: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(aspect).lower()).strip("_")
