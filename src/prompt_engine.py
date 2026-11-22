"""Match judge: deterministic local rules + optional LLM for UNCERTAIN pairs.

* ``local_judge``  - rule-based, always available, no network.
* ``llm_judge``    - calls the Anthropic Messages API (needs ANTHROPIC_API_KEY)
                     and is only used for pairs the local rules cannot decide.
Every decision is validated, evidence is verified against the real features and
the judge is evaluated against human review labels.
"""
import json
import os
from dataclasses import asdict, dataclass, field

import pandas as pd

from src.config import (
    EXPLANATIONS_PATH, JUDGE_EVAL_PATH, JUDGE_PATH, LLM_MODEL, MATCHES_PATH,
    PROMPT_PATH, REPORT_DIR,
)

PROMPTS = {
    "Basic Prompt": """
    Compare two business records.
    Decide whether they represent the same business.
    Return MATCH, NON_MATCH or UNCERTAIN.
    """,
    "Evidence Based Prompt": """
    You are a business entity resolution assistant.

    Compare the two business records using:
    1. Company name similarity
    2. Address similarity
    3. Website domain, phone and email
    4. Country and city

    Use only the supplied evidence.
    Do not assume that similar names always mean the same legal entity.
    Chains and franchises can share names and websites.

    Return MATCH, NON_MATCH or UNCERTAIN.
    Explain your decision.
    """,
    "Structured JSON Prompt": """
    You are an entity resolution expert.

    Analyse the supplied business records carefully.

    Rules:
    - Consider all available business attributes.
    - Do not invent missing information.
    - Identify supporting and conflicting evidence.
    - Return UNCERTAIN when evidence is insufficient.
    - Never make an unsupported claim.

    Return ONLY valid JSON with keys:
    decision (MATCH | NON_MATCH | UNCERTAIN), confidence (0-1),
    evidence (list of strings), explanation (string).
    """,
}

DECISIONS = ("MATCH", "NON_MATCH", "UNCERTAIN")


@dataclass
class MatchDecision:
    decision: str
    confidence: float
    evidence: list = field(default_factory=list)
    explanation: str = ""
    requires_human_review: bool = True
    source: str = "local"

    def __post_init__(self):
        self.decision = str(self.decision).upper()
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        self.confidence = float(self.confidence)
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.evidence, list):
            raise ValueError("evidence must be a list")


def _val(row, key):
    value = row.get(key, 0)
    return 0 if pd.isna(value) else value


def evidence_list(row):
    ev = [f"Name similarity: {_val(row, 'name_similarity') * 100:.1f}%",
          f"Name token overlap: {_val(row, 'name_token_set') * 100:.1f}%",
          f"Address similarity: {_val(row, 'address_similarity') * 100:.1f}%"]
    for key, yes, no in (
            ("same_website", "Website domains match", "Website domains do not match or are unavailable"),
            ("same_phone", "Phone numbers match", "Phone numbers do not match or are unavailable"),
            ("same_email", "Emails match", "Emails do not match or are unavailable"),
            ("same_country", "Countries match", "Countries differ or are unavailable"),
            ("same_city", "Cities match", "Cities differ or are unavailable")):
        ev.append(yes if _val(row, key) else no)
    return ev


def local_judge(row):
    """Deterministic rules. ``row`` is a mapping/Series of pair features."""
    name = _val(row, "name_similarity")
    token = _val(row, "name_token_set")
    strong_id = bool(_val(row, "same_website") or _val(row, "same_phone")
                     or _val(row, "same_email"))
    same_place = bool(_val(row, "same_city") or _val(row, "geo_close")
                      or _val(row, "address_similarity") >= 0.80)
    country_conflict = bool(_val(row, "country_conflict"))

    if name >= 0.95 and strong_id and same_place and not country_conflict:
        decision, conf = "MATCH", 0.95
        why = "Identical names, a confirming website/phone/email and the same location."
    elif (country_conflict and not strong_id) or (token < 0.50 and not strong_id):
        decision, conf = "NON_MATCH", 0.90
        why = "Names or countries conflict and no contact detail supports a match."
    else:
        decision, conf = "UNCERTAIN", 0.60
        why = "Available evidence is insufficient for a reliable automatic decision."
    return MatchDecision(decision, conf, evidence_list(row), why)


def verify_evidence(row, result):
    """Check that every claim in the decision is backed by the real features."""
    claims = {
        "Website domains match": "same_website", "Phone numbers match": "same_phone",
        "Emails match": "same_email", "Countries match": "same_country",
        "Cities match": "same_city",
    }
    issues = [f"Unsupported claim: {text}" for text, key in claims.items()
              if text in result.evidence and not _val(row, key)]
    return not issues, issues


# ------------------------------------------------------------------- LLM
def build_llm_message(row):
    payload = {
        "record_A": {"name": row.get("company_A"), "city": row.get("city_A"),
                     "country": row.get("country_A")},
        "record_B": {"name": row.get("company_B"), "city": row.get("city_B"),
                     "country": row.get("country_B")},
        "evidence": evidence_list(row),
        "rule_based_score": _val(row, "match_score"),
    }
    return PROMPTS["Structured JSON Prompt"] + "\n\nPair:\n" + json.dumps(payload, default=str)


def call_llm(message, timeout=60):
    """Low-level API call (isolated so it can be mocked in tests)."""
    import requests
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": os.environ.get("DATATWIN_LLM_MODEL", LLM_MODEL),
              "max_tokens": 500,
              "messages": [{"role": "user", "content": message}]},
        timeout=timeout)
    response.raise_for_status()
    return "".join(b.get("text", "") for b in response.json()["content"])


def parse_llm_output(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("No JSON object in LLM output")
    data = json.loads(text[start:end + 1])
    return MatchDecision(data["decision"], data["confidence"],
                         list(data.get("evidence", [])), str(data.get("explanation", "")),
                         True, "llm")


def llm_judge(row, caller=call_llm):
    return parse_llm_output(caller(build_llm_message(row)))


# --------------------------------------------------------------- pipeline
def evaluate_against_humans(results, db_path=None):
    """Agreement of judge decisions with human APPROVED/REJECTED labels."""
    from src.review_manager import get_reviewed_pairs
    from src.config import DB_PATH
    reviewed = get_reviewed_pairs(db_path or DB_PATH)
    if reviewed.empty or results.empty:
        return {}
    merged = results.merge(reviewed[["record_id_A", "record_id_B", "decision"]],
                           on=["record_id_A", "record_id_B"], suffixes=("", "_human"))
    decided = merged[merged["decision"] != "UNCERTAIN"]
    correct = ((decided["decision"] == "MATCH") & (decided["decision_human"] == "APPROVED")) | \
              ((decided["decision"] == "NON_MATCH") & (decided["decision_human"] == "REJECTED"))
    return {
        "reviewed_pairs_judged": int(len(merged)),
        "judge_decided": int(len(decided)),
        "judge_uncertain": int(len(merged) - len(decided)),
        "agreement_on_decided": round(float(correct.mean()), 3) if len(decided) else None,
        "coverage": round(len(decided) / len(merged), 3) if len(merged) else None,
    }


def run_prompt_engine(use_llm=False, limit=1000, llm_limit=50):
    matches = pd.read_csv(MATCHES_PATH)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROMPT_PATH, "w", encoding="utf-8") as file:
        for name, prompt in PROMPTS.items():
            file.write(f"\n{'=' * 60}\n{name}\n{'=' * 60}\n{prompt}\n")

    # judge the top pairs + every human-reviewed pair (so we can evaluate)
    from src.config import DB_PATH
    from src.review_manager import get_reviewed_pairs
    targets = matches.head(limit)
    if DB_PATH.exists():
        reviewed = get_reviewed_pairs(DB_PATH)[["record_id_A", "record_id_B"]]
        extra = matches.merge(reviewed, on=["record_id_A", "record_id_B"])
        targets = pd.concat([targets, extra]).drop_duplicates(
            ["record_id_A", "record_id_B"])

    results, llm_calls = [], 0
    for row in targets.to_dict("records"):
        decision = local_judge(row)
        if use_llm and decision.decision == "UNCERTAIN" and llm_calls < llm_limit:
            try:
                decision = llm_judge(row)
                llm_calls += 1
            except Exception as error:  # network/parse errors: keep the local result
                print("LLM judge skipped:", error)
                use_llm = False
        verified, issues = verify_evidence(row, decision)
        results.append({
            "record_id_A": row["record_id_A"], "record_id_B": row["record_id_B"],
            "company_A": row["company_A"], "company_B": row["company_B"],
            "decision": decision.decision, "confidence": decision.confidence,
            "source": decision.source, "evidence": json.dumps(decision.evidence),
            "explanation": decision.explanation, "evidence_verified": verified,
            "verification_issues": json.dumps(issues),
            "requires_human_review": decision.requires_human_review,
        })
    out = pd.DataFrame(results)
    out.to_csv(JUDGE_PATH, index=False)

    evaluation = evaluate_against_humans(out) if DB_PATH.exists() else {}
    JUDGE_EVAL_PATH.write_text(json.dumps(evaluation, indent=2))

    print("\n========== DATATWIN JUDGE REPORT ==========")
    print("Pairs judged:", len(out), "| LLM calls:", llm_calls)
    print(out["decision"].value_counts().to_string())
    print("Evidence verified:", int(out["evidence_verified"].sum()), "/", len(out))
    if evaluation:
        print("Agreement with human labels:", evaluation)
    print("Saved:", JUDGE_PATH)
    return out


if __name__ == "__main__":
    run_prompt_engine(use_llm=bool(os.environ.get("ANTHROPIC_API_KEY")))
