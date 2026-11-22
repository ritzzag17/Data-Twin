"""Entity clusters (connected components) and golden-record merging.

If A=B and B=C then {A, B, C} is one real-world business. Clusters can be
built from human-APPROVED pairs (safe) or from high-confidence ML predictions.
"""
import json
from collections import Counter

import pandas as pd

from src.config import (
    CLUSTER_MAX_SIZE, CLUSTER_PROB_THRESHOLD, CLUSTER_SUMMARY_PATH,
    CLUSTERS_PATH, DB_PATH, GOLDEN_PATH, PREDICTIONS_PATH, RAW_PATH,
    REPORT_DIR,
)
from src.text_utils import is_missing

GOLDEN_FIELDS = ["company_name", "website_domain", "main_country", "main_city",
                 "main_address_raw_text", "primary_phone", "primary_email",
                 "year_founded"]


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:       # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_clusters(pairs):
    """pairs: iterable of (id_a, id_b). Returns {record_id: cluster_id}."""
    uf = UnionFind()
    for a, b in pairs:
        uf.union(int(a), int(b))
    return {rid: uf.find(rid) for rid in list(uf.parent)}


def golden_record(members):
    """Merge a list of record dicts: most frequent non-empty value per field;
    ties are broken by the longest (most informative) value."""
    merged = {}
    for fld in GOLDEN_FIELDS:
        values = [str(m[fld]).strip() for m in members
                  if fld in m and not is_missing(m[fld])]
        if not values:
            merged[fld] = None
            continue
        counts = Counter(v.lower() for v in values)
        top = max(counts.values())
        candidates = [v for v in values if counts[v.lower()] == top]
        merged[fld] = max(candidates, key=len)
    return merged


def pairs_from_humans(db_path=DB_PATH):
    from src.review_manager import get_reviewed_pairs
    reviewed = get_reviewed_pairs(db_path)
    approved = reviewed[reviewed["decision"] == "APPROVED"]
    return list(zip(approved["record_id_A"], approved["record_id_B"]))


def pairs_from_predictions(threshold=CLUSTER_PROB_THRESHOLD, path=PREDICTIONS_PATH):
    pred = pd.read_csv(path)
    strong = pred[(pred["ensemble_probability"] >= threshold)
                  & (pred["match_score"] >= 0.90)]
    return list(zip(strong["record_id_A"], strong["record_id_B"]))


def build_cluster_tables(raw_df, pairs, source, max_size=CLUSTER_MAX_SIZE):
    mapping = build_clusters(pairs)
    if not mapping:
        return pd.DataFrame(), pd.DataFrame(), {"source": source, "clusters": 0}
    members = pd.DataFrame({"record_id": list(mapping), "cluster_id": list(mapping.values())})
    sizes = members.groupby("cluster_id")["record_id"].transform("count")
    members["cluster_size"] = sizes
    members["needs_review"] = sizes > max_size
    members["source"] = source

    golden_rows = []
    for cid, grp in members.groupby("cluster_id"):
        records = raw_df.loc[grp["record_id"], [c for c in GOLDEN_FIELDS if c in raw_df.columns]
                             ].to_dict("records")
        golden_rows.append({"cluster_id": cid, "member_count": len(grp),
                            "member_ids": ",".join(map(str, sorted(grp["record_id"]))),
                            "needs_review": len(grp) > max_size,
                            **golden_record(records)})
    golden = pd.DataFrame(golden_rows)
    size_counts = members.drop_duplicates("cluster_id")["cluster_size"]
    summary = {
        "source": source, "clusters": int(members["cluster_id"].nunique()),
        "records_in_clusters": int(len(members)),
        "largest_cluster": int(size_counts.max()),
        "clusters_needing_review": int((size_counts > max_size).sum()),
        "duplicates_removed_if_merged": int(len(members) - members["cluster_id"].nunique()),
    }
    return members, golden, summary


def main(source="humans"):
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    if source == "predictions":
        pairs = pairs_from_predictions()
    else:
        pairs = pairs_from_humans()
    members, golden, summary = build_cluster_tables(raw, pairs, source)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    members.to_csv(CLUSTERS_PATH, index=False)
    golden.to_csv(GOLDEN_PATH, index=False)
    CLUSTER_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n========== ENTITY CLUSTER REPORT ==========")
    print(json.dumps(summary, indent=2))
    print("Saved:", CLUSTERS_PATH, "and", GOLDEN_PATH)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "humans")
