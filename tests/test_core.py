"""Core tests:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import review_manager as rm  # noqa: E402
from src.blocking import build_candidate_pairs  # noqa: E402
from src.clustering import build_cluster_tables, build_clusters, golden_record  # noqa: E402
from src.duplicate_detection import find_candidates  # noqa: E402
from src.features import match_score  # noqa: E402
from src.prompt_engine import MatchDecision, local_judge, llm_judge, verify_evidence  # noqa: E402
from src.text_utils import (  # noqa: E402
    normalize_domain, normalize_name, normalize_phone, prepare_frame,
    sim_jaro_winkler, sim_ratio, sim_token_set,
)
from src.train_model import build_models, cross_validate  # noqa: E402
from src.trust_score import compute_trust  # noqa: E402

BASE = {"name_similarity": 1.0, "name_token_set": 1.0, "address_similarity": 0.0,
        "same_website": 0, "same_phone": 0, "same_email": 0, "same_city": 1,
        "same_country": 1, "geo_close": 0, "country_conflict": 0, "city_conflict": 0}


def sample_df():
    return pd.DataFrame({
        "company_name": ["Acme Widgets GmbH", "ACME Widgets", "Zeta Foods Ltd", "Acme Widgets Inc", None],
        "website_domain": ["https://www.acme.com/", "acme.com", "zeta.org", "acme.com", "acme.com"],
        "main_country": ["Germany", "Germany", "France", "United States", "Germany"],
        "main_city": ["Berlin", "Berlin", "Paris", "Austin", "Berlin"],
        "main_address_raw_text": ["1 Main St, Berlin", "1 Main St, Berlin", "5 Rue X", None, None],
        "primary_phone": [4930123456.0, "+49 30 123456", None, None, None],
        "primary_email": ["a@acme.com", None, "bad-email", None, None],
    })


# ---------------------------------------------------------------- text utils
def test_name_normalisation_strips_suffix_and_accents():
    assert normalize_name("Gerüstbau Burbach GmbH") == "gerustbau burbach"
    assert normalize_name("Acme, Inc.") == "acme"
    assert normalize_name("GmbH") == "gmbh"          # never strip to empty
    assert normalize_name(None) == ""


def test_phone_and_domain_normalisation():
    assert normalize_phone(4930123456.0) == normalize_phone("+49 (30) 123456")
    assert normalize_phone("123") == ""
    assert normalize_domain("HTTPS://www.Acme.com/path?x=1") == "acme.com"


def test_similarity_bounds():
    assert sim_ratio("abc", "abc") == 1.0
    assert sim_ratio("", "abc") == 0.0
    assert 0 <= sim_jaro_winkler("martha", "marhta") <= 1
    assert sim_token_set("acme widgets", "widgets acme berlin") > 0.9


# ----------------------------------------------------------------- scoring
def test_conflicts_lower_the_score():
    clean = match_score(BASE)
    assert match_score({**BASE, "country_conflict": 1, "same_country": 0}) < clean
    assert match_score({**BASE, "city_conflict": 1, "same_city": 0}) < clean
    assert 0 <= match_score(BASE) <= 1


def test_strong_identifiers_raise_the_score():
    assert match_score({**BASE, "same_website": 1, "same_phone": 1}) > match_score(BASE)


# ---------------------------------------------------------------- blocking
def test_blocking_and_find_candidates():
    pairs, stats = build_candidate_pairs(prepare_frame(sample_df()))
    assert (0, 1) in pairs and (0, 3) in pairs     # same domain / same name
    assert all(a < b for a, b in pairs)
    matches, _ = find_candidates(sample_df())
    assert not matches.empty
    top = matches.iloc[0]
    assert {top.record_id_A, top.record_id_B} == {0, 1}
    assert top.same_website == 1 and top.same_phone == 1


def test_oversized_blocks_are_subblocked_not_dropped():
    n = 30
    df = pd.DataFrame({
        "company_name": ["Shared Name"] * n,
        "website_domain": [f"d{i}.com" for i in range(n)],
        "main_country": ["Spain"] * n,
        "main_city": ["Madrid"] * 15 + ["Sevilla"] * 15,
    })
    pairs, stats = build_candidate_pairs(prepare_frame(df), max_block_size=20)
    assert (0, 1) in pairs and (15, 16) in pairs and (0, 15) not in pairs
    assert stats["oversized_blocks"].sum() > 0


# -------------------------------------------------------------- clustering
def test_clusters_and_golden_record():
    mapping = build_clusters([(1, 2), (2, 3), (10, 11)])
    assert mapping[1] == mapping[2] == mapping[3] != mapping[10]
    golden = golden_record([
        {"company_name": "Acme", "website_domain": "acme.com", "main_city": None},
        {"company_name": "Acme", "website_domain": "acme.com", "main_city": "Berlin"},
        {"company_name": "ACME Ltd", "website_domain": None, "main_city": "Berlin"},
    ])
    assert golden["company_name"].lower() == "acme" and golden["main_city"] == "Berlin"
    members, golden_df, summary = build_cluster_tables(sample_df(), [(0, 1), (1, 3)], "test")
    assert summary["clusters"] == 1 and summary["duplicates_removed_if_merged"] == 2


# ----------------------------------------------------------- review manager
def test_review_flow_with_audit_trail(tmp_path):
    db = tmp_path / "t.db"
    rm.initialize_database(db)
    matches = pd.DataFrame({"record_id_A": [1, 2, 3], "record_id_B": [4, 5, 6],
                            "company_A": list("abc"), "company_B": list("xyz"),
                            "match_score": [0.9, 0.8, 0.7], "match_category": ["High"] * 3})
    assert rm.import_matches(matches, db) == 3
    rm.set_uncertainty(pd.DataFrame({"record_id_A": [1, 2, 3], "record_id_B": [4, 5, 6],
                                     "uncertainty": [0.1, 0.9, 0.5]}), db)
    nxt = rm.get_next_pair("uncertain", db_path=db)
    assert nxt["record_id_A"] == 2                      # most uncertain first
    assert rm.get_next_pair("uncertain", skip_ids=[nxt["id"]], db_path=db)["record_id_A"] == 3

    rm.save_decision(int(nxt["id"]), "APPROVED", "tester", "looks right", db)
    assert rm.get_decision_counts(db) == {"PENDING": 2, "APPROVED": 1, "REJECTED": 0}
    # re-import must not overwrite a human decision
    rm.import_matches(matches.assign(match_score=0.1), db)
    assert rm.get_decision_counts(db)["APPROVED"] == 1
    assert rm.undo_last_decision("tester", db) == nxt["id"]
    assert rm.get_decision_counts(db)["APPROVED"] == 0
    assert len(rm.get_review_history(db_path=db)) == 2


def test_invalid_decision_rejected(tmp_path):
    db = tmp_path / "t.db"
    rm.initialize_database(db)
    try:
        rm.save_decision(1, "MAYBE", db_path=db)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_old_database_is_migrated(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE match_reviews (id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id_A INTEGER NOT NULL, record_id_B INTEGER NOT NULL, company_A TEXT, company_B TEXT,
        match_score REAL, match_category TEXT, decision TEXT DEFAULT 'PENDING',
        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(record_id_A, record_id_B))""")
    con.execute("INSERT INTO match_reviews (record_id_A, record_id_B, decision) VALUES (1, 2, 'APPROVED')")
    con.commit(); con.close()
    rm.initialize_database(db)
    assert rm.get_decision_counts(db)["APPROVED"] == 1


# -------------------------------------------------------------------- judge
def test_judge_rules_and_verification():
    match_row = {**BASE, "same_website": 1, "address_similarity": 0.9}
    assert local_judge(match_row).decision == "MATCH"
    far = {**BASE, "name_similarity": 0.2, "name_token_set": 0.2, "same_city": 0, "same_country": 0}
    assert local_judge(far).decision == "NON_MATCH"
    assert local_judge({**BASE}).decision == "UNCERTAIN"
    decision = local_judge(match_row)
    assert verify_evidence(match_row, decision)[0]
    assert not verify_evidence({**match_row, "same_website": 0}, decision)[0]


def test_decision_validation_and_llm_parsing():
    try:
        MatchDecision("MAYBE", 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid decision accepted")
    fake = lambda msg: 'Sure! {"decision": "match", "confidence": 0.8, "evidence": ["x"], "explanation": "y"}'
    result = llm_judge({**BASE, "company_A": "A", "company_B": "B"}, caller=fake)
    assert result.decision == "MATCH" and result.source == "llm" and result.requires_human_review


# --------------------------------------------------------------- trust + ML
def test_trust_score_components():
    scores, issues = compute_trust(sample_df(), duplicate_ids={0, 1})
    assert scores["trust_score"].between(0, 100).all()
    assert scores.loc[0, "trust_score"] > scores.loc[4, "trust_score"]
    assert "Possible Duplicate" in set(issues["issue"])
    assert "Invalid Email Format" in set(issues["issue"])


def test_cross_validation_runs_and_reports_baseline():
    rng = np.random.default_rng(0)
    n = 60
    y = pd.Series(rng.integers(0, 2, n))
    X = pd.DataFrame({"f1": y + rng.normal(0, 0.5, n), "f2": rng.normal(0, 1, n)})
    X["match_score"] = 0.5 + 0.4 * y
    summary, folds = cross_validate(build_models(), X, y, repeats=2)
    assert folds >= 2 and "rule_baseline" in summary
    assert summary["random_forest"]["roc_auc"]["mean"] > 0.7
