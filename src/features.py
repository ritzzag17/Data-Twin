"""Pair features and rule-based match score."""
import math

from src.config import (
    CITY_CONFLICT_PENALTY, COUNTRY_CONFLICT_PENALTY, GEO_CLOSE_KM,
    HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, SCORE_WEIGHTS,
)
from src.text_utils import (
    haversine_km, sim_jaro_winkler, sim_ratio, sim_token_set, sim_token_sort,
)

RECORD_COLUMNS = [
    "name_tokens", "name_key", "domain_key", "phone_key", "email_key",
    "country_key", "city_key", "address_key", "street_no", "lat", "lon",
    "industry_key",
]


def _same(x, y):
    return int(bool(x) and x == y)


def pair_features(a, b):
    """Compute evidence features for two prepared records (namedtuples)."""
    both_city = bool(a.city_key) and bool(b.city_key)
    both_country = bool(a.country_key) and bool(b.country_key)

    distance = math.nan
    if not any(math.isnan(v) for v in (a.lat, a.lon, b.lat, b.lon)):
        distance = haversine_km(a.lat, a.lon, b.lat, b.lon)

    return {
        "name_similarity": sim_ratio(a.name_key, b.name_key),
        "name_token_set": sim_token_set(a.name_tokens, b.name_tokens),
        "name_jaro_winkler": sim_jaro_winkler(a.name_key, b.name_key),
        "address_similarity": sim_token_sort(a.address_key, b.address_key),
        "same_website": _same(a.domain_key, b.domain_key),
        "same_phone": _same(a.phone_key, b.phone_key),
        "same_email": _same(a.email_key, b.email_key),
        "same_city": int(both_city and a.city_key == b.city_key),
        "same_country": int(both_country and a.country_key == b.country_key),
        "same_street_number": _same(a.street_no, b.street_no),
        "geo_distance_km": None if math.isnan(distance) else round(distance, 3),
        "geo_close": int(not math.isnan(distance) and distance <= GEO_CLOSE_KM),
        "same_industry": _same(a.industry_key, b.industry_key),
        "country_conflict": int(both_country and a.country_key != b.country_key),
        "city_conflict": int(both_city and a.city_key != b.city_key),
    }


def match_score(f):
    """Weighted evidence score in [0, 1] with penalties for conflicts."""
    w = SCORE_WEIGHTS
    # Blend of character-level and token-level similarity. When one name's
    # tokens are contained in the other ("X" vs "X Co. Tokyo Office") the
    # token-set score is 1.0, so give that partial credit.
    name = max((f["name_similarity"] + f["name_token_set"]) / 2,
               0.85 * f["name_token_set"])
    address = 1.0 if f["geo_close"] else f["address_similarity"]

    score = (
        w["name"] * name
        + w["website"] * f["same_website"]
        + w["phone"] * f["same_phone"]
        + w["email"] * f["same_email"]
        + w["address"] * address
        + w["city"] * f["same_city"]
        + w["country"] * f["same_country"]
    )
    if f["country_conflict"]:
        score -= COUNTRY_CONFLICT_PENALTY
    if f["city_conflict"] and not f["geo_close"] and f["address_similarity"] < 0.5:
        score -= CITY_CONFLICT_PENALTY
    return round(min(max(score, 0.0), 1.0), 3)


def match_category(score):
    if score >= HIGH_CONFIDENCE:
        return "High Confidence"
    if score >= MEDIUM_CONFIDENCE:
        return "Medium Confidence"
    return "Low Confidence"
