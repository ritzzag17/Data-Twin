"""Step: Data Trust Score = completeness + validity + uniqueness.

Vectorised (no row loops). The score is still *not* a measure of factual
accuracy - it measures how complete, well-formed and non-duplicated a record is.
"""
import pandas as pd

from src.config import (
    FIELD_WEIGHTS, HIGH_CONFIDENCE, HIGH_QUALITY, MATCHES_PATH, MEDIUM_QUALITY,
    RAW_PATH, REPAIR_PATH, REPORT_DIR, TRUST_COMPONENT_WEIGHTS, TRUST_PATH,
)
from src.text_utils import normalize_domain, normalize_email, normalize_phone

VALIDATORS = {
    "primary_email": ("Invalid Email Format", normalize_email,
                      "Verify the email address before updating it."),
    "website_domain": ("Invalid Domain Format",
                       lambda v: normalize_domain(v) if "." in normalize_domain(v) else "",
                       "Verify and normalise the website domain."),
    "primary_phone": ("Invalid Phone Number", normalize_phone,
                      "Verify the phone number (expected 7+ digits)."),
}


def _missing_mask(series):
    return series.isna() | (series.astype(str).str.strip() == "")


def compute_trust(df, duplicate_ids=None):
    """Return (scores_df, issues_df)."""
    fields = [f for f in FIELD_WEIGHTS if f in df.columns]
    missing = pd.DataFrame({f: _missing_mask(df[f]) for f in fields})
    present = ~missing

    total_weight = sum(FIELD_WEIGHTS.values())
    completeness = sum(FIELD_WEIGHTS[f] * present[f] for f in fields) / total_weight * 100

    # Validity: share of validatable fields that are present AND well-formed
    invalid = pd.DataFrame(index=df.index)
    for field, (_, normaliser, _) in VALIDATORS.items():
        if field in df.columns:
            invalid[field] = present[field] & (df[field].map(normaliser) == "")
    validatable = present[[f for f in invalid.columns]]
    n_validatable = validatable.sum(axis=1)
    validity = ((validatable & ~invalid).sum(axis=1) / n_validatable.where(n_validatable > 0)) * 100

    # Uniqueness: records in a high-confidence duplicate pair are penalised
    if duplicate_ids is not None:
        uniqueness = pd.Series(100.0, index=df.index)
        uniqueness.loc[uniqueness.index.isin(duplicate_ids)] = 40.0
    else:
        uniqueness = pd.Series(float("nan"), index=df.index)

    weights = TRUST_COMPONENT_WEIGHTS
    components = {"completeness": completeness, "validity": validity, "uniqueness": uniqueness}
    num = sum(weights[k] * v.fillna(0) for k, v in components.items())
    den = sum(weights[k] * v.notna() for k, v in components.items())
    trust = (num / den).round(2)

    category = pd.cut(trust, [-1, MEDIUM_QUALITY - 1e-9, HIGH_QUALITY - 1e-9, 101],
                      labels=["Low Quality", "Medium Quality", "High Quality"])

    issues_per_row = missing.sum(axis=1) + invalid.sum(axis=1)
    scores = pd.DataFrame({
        "record_id": df.index,
        "company_name": df.get("company_name"),
        "completeness_score": completeness.round(2),
        "validity_score": validity.round(2),
        "uniqueness_score": uniqueness.round(2),
        "trust_score": trust,
        "quality_category": category.astype(str),
        "issues_count": issues_per_row,
    })

    frames = []
    for field in fields:
        severity = "High" if FIELD_WEIGHTS[field] >= 15 else "Medium"
        idx = df.index[missing[field]]
        frames.append(pd.DataFrame({
            "record_id": idx, "field": field, "issue": "Missing Value",
            "severity": severity,
            "suggestion": "Enrich this field using a verified business source."}))
        text = df[field].astype(str)
        idx = df.index[present[field] & (text != text.str.strip())]
        frames.append(pd.DataFrame({
            "record_id": idx, "field": field, "issue": "Extra Whitespace",
            "severity": "Low",
            "suggestion": "Remove leading or trailing whitespace."}))
    for field, (label, _, suggestion) in VALIDATORS.items():
        if field in invalid:
            idx = df.index[invalid[field]]
            frames.append(pd.DataFrame({
                "record_id": idx, "field": field, "issue": label,
                "severity": "Medium", "suggestion": suggestion}))
    if duplicate_ids is not None:
        idx = df.index[df.index.isin(duplicate_ids)]
        frames.append(pd.DataFrame({
            "record_id": idx, "field": "record", "issue": "Possible Duplicate",
            "severity": "High",
            "suggestion": "Review in Human Review and merge into one record."}))

    issues = pd.concat(frames, ignore_index=True)
    names = df["company_name"] if "company_name" in df.columns else None
    if names is not None:
        issues.insert(1, "company_name", issues["record_id"].map(names))
    return scores, issues.sort_values(["record_id", "severity"]).reset_index(drop=True)


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    duplicate_ids = None
    if MATCHES_PATH.exists():
        matches = pd.read_csv(MATCHES_PATH, usecols=["record_id_A", "record_id_B", "match_score"])
        strong = matches[matches["match_score"] >= HIGH_CONFIDENCE]
        duplicate_ids = set(strong["record_id_A"]) | set(strong["record_id_B"])
    else:
        print("potential_matches.csv not found - uniqueness component skipped.")

    scores, issues = compute_trust(df, duplicate_ids)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(TRUST_PATH, index=False)
    issues.to_csv(REPAIR_PATH, index=False)

    print("\n========== DATATWIN DATA TRUST REPORT ==========")
    print("Records:", f"{len(scores):,}")
    print("Average trust score:", round(scores["trust_score"].mean(), 2))
    print(scores["quality_category"].value_counts().to_string())
    print("\nIssues:", f"{len(issues):,}")
    print(issues["issue"].value_counts().to_string())
    print(f"\nSaved: {TRUST_PATH}\nSaved: {REPAIR_PATH}")


if __name__ == "__main__":
    main()
