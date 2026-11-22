"""Evidence-based explanations (supporting AND conflicting evidence) per pair."""
import pandas as pd

from src.config import EXPLANATIONS_PATH, MATCHES_PATH, REPORT_DIR


def _pct(value):
    return f"{float(value) * 100:.0f}%"


def explain_pair(row):
    """Return dict with evidence list, supporting/conflicting evidence, explanation."""
    g = lambda k: row.get(k, 0) if pd.notna(row.get(k, 0)) else 0
    supporting, conflicting = [], []

    name = max(g("name_similarity"), g("name_token_set"))
    if name >= 0.95:
        supporting.append("near-identical company names")
    elif name >= 0.85:
        supporting.append("highly similar company names")
    elif name < 0.60:
        conflicting.append("company names are dissimilar")

    for key, label in (("same_website", "same website domain"),
                       ("same_phone", "same phone number"),
                       ("same_email", "same email address"),
                       ("same_street_number", "same street number")):
        if g(key):
            supporting.append(label)
    if g("geo_close"):
        supporting.append("locations are within ~150 m")
    if g("address_similarity") >= 0.80:
        supporting.append("similar addresses")
    elif 0 < g("address_similarity") < 0.40:
        conflicting.append("addresses differ")
    if g("same_city"):
        supporting.append("same city")
    if g("same_industry"):
        supporting.append("same industry")
    if g("country_conflict"):
        conflicting.append("different countries")
    if g("city_conflict") and not g("geo_close"):
        conflicting.append("different cities")

    evidence = [
        f"Name similarity: {_pct(g('name_similarity'))}",
        f"Name token overlap: {_pct(g('name_token_set'))}",
        f"Address similarity: {_pct(g('address_similarity'))}",
        "Website matches" if g("same_website") else "Website does not match / unavailable",
        "Phone matches" if g("same_phone") else "Phone does not match / unavailable",
        "City matches" if g("same_city") else "City differs / unavailable",
        "Country matches" if g("same_country") else "Country differs / unavailable",
    ]

    if supporting:
        text = "Potential match because of: " + ", ".join(supporting) + "."
    else:
        text = "Limited supporting evidence."
    if conflicting:
        text += " Conflicting evidence: " + ", ".join(conflicting) + "."
    if not (g("same_website") or g("same_phone") or g("same_email")):
        text += " No website, phone or email confirms the match."

    return {
        "evidence": " | ".join(evidence),
        "supporting_evidence": "; ".join(supporting),
        "conflicting_evidence": "; ".join(conflicting),
        "explanation": text,
        "review_recommendation": "Human review required",
    }


def generate_explanations(matches_path=MATCHES_PATH, output_path=EXPLANATIONS_PATH):
    matches = pd.read_csv(matches_path)
    explanations = pd.DataFrame([explain_pair(r) for r in matches.to_dict("records")])
    out = pd.concat([matches[["record_id_A", "record_id_B", "company_A", "company_B",
                              "match_score", "match_category"]], explanations], axis=1)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("\n========== MATCH EXPLANATION REPORT ==========")
    print("Total explanations:", f"{len(out):,}")
    for _, row in out.head(3).iterrows():
        print("\n", row["company_A"], "<-->", row["company_B"], f"({row['match_score']})")
        print(" ", row["explanation"])
    print("\nSaved:", output_path)
    return out


if __name__ == "__main__":
    generate_explanations()
