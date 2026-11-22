"""Normalisation helpers and string-similarity wrappers.

Uses RapidFuzz when installed (fast). Falls back to a pure-Python
implementation so the code still runs (slower) without it.
"""
import math
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

try:  # pragma: no cover - depends on environment
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz.distance import JaroWinkler as _JW
    HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    HAS_RAPIDFUZZ = False

LEGAL_SUFFIXES = {
    "gmbh", "ag", "kg", "ug", "ltd", "limited", "llc", "llp", "lp", "inc",
    "incorporated", "corp", "corporation", "co", "company", "plc", "sa",
    "sas", "sarl", "srl", "spa", "sl", "bv", "nv", "ab", "as", "oy", "oyj",
    "aps", "pty", "pvt", "ltda", "eireli", "ooo", "zao", "kk", "sdn", "bhd",
    "pte", "ev", "mbh", "gbr", "ohg", "cv", "se", "sp", "zoo",
}
STOP_TOKENS = {"the", "a", "an", "and", "of", "la", "le", "el", "de", "der",
               "die", "das", "los", "las", "du", "des", "van", "von"}


# ------------------------------------------------------------------ basics
def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def clean(value):
    """Lower-case, trimmed string; '' for missing values."""
    return "" if is_missing(value) else str(value).strip().lower()


def name_tokens(value):
    """Tokens of a company name without punctuation, accents, legal suffixes."""
    text = strip_accents(clean(value))
    tokens = re.findall(r"[^\W_]+", text)
    stripped = list(tokens)
    while len(stripped) > 1 and stripped[-1] in LEGAL_SUFFIXES:
        stripped.pop()
    return stripped


def normalize_name(value):
    """'Acme Corp. GmbH' -> 'acme corp' (tokens joined by a space)."""
    return " ".join(name_tokens(value))


def name_key(value):
    """Name with no spaces at all, used for character-level similarity."""
    return "".join(name_tokens(value))


def normalize_domain(value):
    text = clean(value)
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return text.split("/")[0].split(":")[0]


def normalize_phone(value):
    """Digits only; last 9 digits to be robust to country-code formats."""
    if is_missing(value):
        return ""
    if isinstance(value, float):
        value = int(value)
    digits = re.sub(r"\D", "", str(value))
    return digits[-9:] if len(digits) >= 7 else ""


def normalize_email(value):
    text = clean(value)
    return text if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", text) else ""


def normalize_address(value):
    text = strip_accents(clean(value))
    return " ".join(re.findall(r"[^\W_]+", text))


def normalize_street_number(number, address=None):
    if not is_missing(number):
        text = str(number).strip().lower()
        return re.sub(r"\.0$", "", text)
    match = re.match(r"^\s*(\d+[a-z]?)\b", clean(address))
    return match.group(1) if match else ""


def naics_prefix(value, length=4):
    if is_missing(value):
        return ""
    try:
        return str(int(float(value)))[:length]
    except (TypeError, ValueError):
        return str(value)[:length]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# -------------------------------------------------------- similarity (0..1)
def _jaro(a, b):
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    window = max(la, lb) // 2 - 1
    ma, mb = [False] * la, [False] * lb
    matches = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - window), min(i + window + 1, lb)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    k = transpositions = 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    t = transpositions / 2
    return (matches / la + matches / lb + (matches - t) / matches) / 3


def _py_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio() * 100


def _py_token_sort(a, b):
    return _py_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def _py_token_set(a, b):
    sa, sb = set(a.split()), set(b.split())
    inter = " ".join(sorted(sa & sb))
    ta = (inter + " " + " ".join(sorted(sa - sb))).strip()
    tb = (inter + " " + " ".join(sorted(sb - sa))).strip()
    if not inter:
        return _py_ratio(ta, tb)
    return max(_py_ratio(inter, ta), _py_ratio(inter, tb), _py_ratio(ta, tb))


def _guard(func):
    def wrapper(a, b):
        if not a or not b:
            return 0.0
        return round(func(a, b), 4)
    return wrapper


if HAS_RAPIDFUZZ:  # pragma: no cover
    sim_ratio = _guard(lambda a, b: _fuzz.ratio(a, b) / 100)
    sim_token_sort = _guard(lambda a, b: _fuzz.token_sort_ratio(a, b) / 100)
    sim_token_set = _guard(lambda a, b: _fuzz.token_set_ratio(a, b) / 100)
    sim_jaro_winkler = _guard(lambda a, b: _JW.similarity(a, b))
else:
    sim_ratio = _guard(lambda a, b: _py_ratio(a, b) / 100)
    sim_token_sort = _guard(lambda a, b: _py_token_sort(a, b) / 100)
    sim_token_set = _guard(lambda a, b: _py_token_set(a, b) / 100)

    def _py_jw(a, b):
        j = _jaro(a, b)
        prefix = 0
        for x, y in zip(a[:4], b[:4]):
            if x != y:
                break
            prefix += 1
        return j + prefix * 0.1 * (1 - j)

    sim_jaro_winkler = _guard(_py_jw)


# ------------------------------------------------------------ frame helper
def prepare_frame(df):
    """Add normalised helper columns. Index becomes the record id (0..n-1)."""
    df = df.reset_index(drop=True).copy()
    for col in ("company_name", "website_domain", "main_country", "main_city",
                "main_address_raw_text", "primary_phone", "primary_email",
                "main_street_number", "main_latitude", "main_longitude",
                "naics_2022_primary_code"):
        if col not in df.columns:
            df[col] = None

    df["name_tokens"] = df["company_name"].map(normalize_name)
    df["name_key"] = df["company_name"].map(name_key)
    df["domain_key"] = df["website_domain"].map(normalize_domain)
    df["phone_key"] = df["primary_phone"].map(normalize_phone)
    df["email_key"] = df["primary_email"].map(normalize_email)
    df["country_key"] = df["main_country"].map(clean)
    df["city_key"] = df["main_city"].map(clean)
    df["address_key"] = df["main_address_raw_text"].map(normalize_address)
    df["street_no"] = [normalize_street_number(n, a) for n, a in
                       zip(df["main_street_number"], df["main_address_raw_text"])]
    df["lat"] = pd.to_numeric(df["main_latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["main_longitude"], errors="coerce")
    df["industry_key"] = df["naics_2022_primary_code"].map(naics_prefix)
    return df
