from __future__ import annotations

import re
import unicodedata
from typing import Any


TEAM_ALIASES = {
    "usa": "united states",
    "u s a": "united states",
    "us": "united states",
    "u s": "united states",
    "united states of america": "united states",
    "czech republic": "czechia",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "ir iran": "iran",
    "iran ir": "iran",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia-herzegovina": "bosnia and herzegovina",
    "congo dr": "dr congo",
    "congo democratic republic": "dr congo",
    "cote d ivoire": "ivory coast",
    "côte d ivoire": "ivory coast",
    "equ guinea": "equatorial guinea",
    "dominican rep": "dominican republic",
    "kyrgyz republic": "kyrgyzstan",
}

TEAM_CODE_TOKENS = {
    "ar", "arg", "au", "aus", "at", "aut", "be", "bel", "br", "bra", "ca", "can",
    "ch", "chi", "cl", "co", "col", "cr", "cro", "cz", "de", "den", "do", "ec",
    "ecu", "eng", "es", "esp", "fr", "fra", "ger", "gh", "gha", "ir", "irn", "it",
    "ita", "jp", "jpn", "kr", "kor", "ma", "mar", "mx", "mex", "nl", "ned", "pt",
    "por", "qa", "qat", "sa", "sau", "sc", "sco", "se", "sen", "sn", "sui", "sv",
    "tr", "tur", "tn", "tun", "uk", "ur", "uru", "us", "usa",
}


def ascii_fold(value: Any) -> str:
    return unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def canonical_person_name(name: Any) -> str:
    text = ascii_fold(name).lower()
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9 .'-]+", " ", text)
    text = text.replace(".", " ")
    return normalize_spaces(text)


def canonical_team_name(name: Any) -> str:
    text = ascii_fold(name).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9 -]+", " ", text)
    text = normalize_spaces(text.replace("-", " "))
    tokens = text.split()
    if len(tokens) > 1 and tokens[0] in TEAM_CODE_TOKENS:
        tokens = tokens[1:]
    if len(tokens) > 1 and tokens[-1] in TEAM_CODE_TOKENS:
        tokens = tokens[:-1]
    text = " ".join(tokens)
    text = normalize_spaces(text)
    return TEAM_ALIASES.get(text, text)


def clean_display_name(value: Any) -> str:
    text = normalize_spaces(str(value or "").replace("\ufeff", ""))
    return text.strip(" ,;")
