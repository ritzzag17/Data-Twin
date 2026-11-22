"""Multi-key blocking to generate candidate pairs without O(n^2) comparisons.

Unlike a single-key approach, blocks that are too large are *sub-blocked*
(e.g. by city) rather than silently dropped, and every skipped record is
reported in the blocking report.
"""
from itertools import combinations

import pandas as pd

from src.config import MAX_BLOCK_SIZE
from src.text_utils import STOP_TOKENS


def _first_token(tokens):
    for token in tokens.split():
        if token not in STOP_TOKENS:
            return token
    return ""


def _block_definitions(df):
    """Return [(name, key_series, sub_key_series_or_None)]."""
    country = df["country_key"]
    first = df["name_tokens"].map(_first_token)
    sorted_tokens = df["name_tokens"].map(lambda t: " ".join(sorted(t.split())))
    multi_token = df["name_tokens"].str.count(" ") >= 1

    def with_country(series, valid):
        keys = series + "|" + country
        return keys.where(valid & (country != ""), "")

    return [
        ("website_domain", df["domain_key"], df["name_key"].str[:6]),
        ("phone", df["phone_key"], None),
        ("email", df["email_key"], None),
        ("name_prefix+country",
         with_country(df["name_key"].str[:5], df["name_key"].str.len() >= 3),
         df["city_key"]),
        ("first_token+country",
         with_country(first, first.str.len() >= 3),
         df["city_key"]),
        ("sorted_tokens+country",
         with_country(sorted_tokens, multi_token),
         df["city_key"]),
    ]


def _emit(ids, pairs):
    for a, b in combinations(sorted(ids), 2):
        pairs.add((a, b))


def build_candidate_pairs(df, max_block_size=MAX_BLOCK_SIZE):
    """Return (set_of_pairs, blocking_stats_dataframe) for a prepared frame."""
    pairs = set()
    stats = []

    for name, keys, subkeys in _block_definitions(df):
        keys = keys[keys != ""]
        keys = keys[keys.duplicated(keep=False)]
        before = len(pairs)
        n_blocks = oversized = skipped = 0

        for _, group in keys.groupby(keys):
            ids = group.index.tolist()
            n_blocks += 1
            if len(ids) <= max_block_size:
                _emit(ids, pairs)
                continue

            oversized += 1
            if subkeys is None:
                skipped += len(ids)
                continue
            sub = subkeys.loc[ids]
            sub = sub[sub != ""]
            sub = sub[sub.duplicated(keep=False)]
            covered = 0
            for _, subgroup in sub.groupby(sub):
                sub_ids = subgroup.index.tolist()
                if len(sub_ids) <= max_block_size:
                    _emit(sub_ids, pairs)
                    covered += len(sub_ids)
            skipped += len(ids) - covered   # records that could not be compared

        stats.append({
            "blocking_key": name,
            "blocks": n_blocks,
            "oversized_blocks": oversized,
            "records_skipped": skipped,
            "new_pairs": len(pairs) - before,
        })

    return pairs, pd.DataFrame(stats)
