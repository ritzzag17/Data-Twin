"""Step 2: generate candidate duplicate pairs (blocking) and score them."""
import time

import pandas as pd

from src.blocking import build_candidate_pairs
from src.config import (
    BLOCKING_REPORT_PATH, MATCHES_PATH, MAX_BLOCK_SIZE, MAX_CANDIDATE_PAIRS,
    MIN_MATCH_SCORE, RAW_PATH, REPORT_DIR,
)
from src.features import RECORD_COLUMNS, match_category, match_score, pair_features
from src.text_utils import prepare_frame


def find_candidates(df, min_score=MIN_MATCH_SCORE, max_block_size=MAX_BLOCK_SIZE,
                    max_pairs=MAX_CANDIDATE_PAIRS, progress=None):
    """Return (matches_df, blocking_stats_df) for a raw business dataframe.

    Record ids are the row positions (0..n-1) of ``df``.
    """
    frame = prepare_frame(df)
    pairs, stats = build_candidate_pairs(frame, max_block_size=max_block_size)

    if len(pairs) > max_pairs:
        raise ValueError(
            f"{len(pairs):,} candidate pairs exceeds the limit of {max_pairs:,}. "
            "Lower MAX_BLOCK_SIZE or use a smaller dataset."
        )

    records = list(frame[RECORD_COLUMNS].itertuples(index=False))
    names = frame["company_name"].tolist()
    countries = frame["main_country"].tolist()
    cities = frame["main_city"].tolist()

    rows = []
    ordered = sorted(pairs)
    total = len(ordered)
    for n, (a, b) in enumerate(ordered, start=1):
        features = pair_features(records[a], records[b])
        score = match_score(features)
        if score >= min_score:
            rows.append({
                "record_id_A": a, "record_id_B": b,
                "company_A": names[a], "company_B": names[b],
                "country_A": countries[a], "country_B": countries[b],
                "city_A": cities[a], "city_B": cities[b],
                **features,
                "match_score": score,
                "match_category": match_category(score),
            })
        if progress and n % 50_000 == 0:
            progress(n, total)

    matches = pd.DataFrame(rows)
    if not matches.empty:
        matches = matches.sort_values("match_score", ascending=False,
                                      kind="stable").reset_index(drop=True)
    stats.attrs["candidate_pairs"] = total
    return matches, stats


def main():
    start = time.time()
    print("Loading business dataset...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Records: {len(df):,}")

    matches, stats = find_candidates(
        df, progress=lambda n, t: print(f"  scored {n:,}/{t:,} pairs"))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_PATH, index=False)
    stats.to_csv(BLOCKING_REPORT_PATH, index=False)

    print("\n========== BLOCKING REPORT ==========")
    print(stats.to_string(index=False))
    print("\n========== DATATWIN MATCHING REPORT ==========")
    print("Candidate pairs compared:", f"{stats.attrs['candidate_pairs']:,}")
    print("Potential matching pairs:", f"{len(matches):,}")
    if not matches.empty:
        print(matches["match_category"].value_counts().to_string())
    print(f"\nSaved: {MATCHES_PATH}\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
