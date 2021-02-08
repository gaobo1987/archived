import re
import unicodedata


def remove_url(s: str) -> str:
    return re.sub(r'http\S+', '', s)


def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def clean(s: str) -> str:
    return strip_accents(remove_url(s)).lower().strip()
