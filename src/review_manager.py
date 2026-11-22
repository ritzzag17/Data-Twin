"""Human-review database (SQLite) with audit trail and active-learning queue.

Compatible with databases created by the original DataTwin: the existing
``match_reviews`` table is migrated in place (new columns are added, existing
decisions are kept).
"""
import sqlite3
from contextlib import closing

import pandas as pd

from src.config import DB_PATH, MATCHES_PATH

VALID_DECISIONS = {"APPROVED", "REJECTED", "PENDING"}


def connect(db_path=DB_PATH):
    db_path = str(db_path)
    return sqlite3.connect(db_path)


def initialize_database(db_path=DB_PATH):
    from pathlib import Path
    Path(str(db_path)).parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(db_path)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS match_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id_A INTEGER NOT NULL,
                record_id_B INTEGER NOT NULL,
                company_A TEXT,
                company_B TEXT,
                match_score REAL,
                match_category TEXT,
                decision TEXT DEFAULT 'PENDING',
                reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(record_id_A, record_id_B)
            )""")
        existing = {row[1] for row in con.execute("PRAGMA table_info(match_reviews)")}
        for column, ddl in (("uncertainty", "REAL"), ("reviewer", "TEXT"),
                            ("review_note", "TEXT")):
            if column not in existing:
                con.execute(f"ALTER TABLE match_reviews ADD COLUMN {column} {ddl}")
        con.execute("""
            CREATE TABLE IF NOT EXISTS review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER NOT NULL,
                decision TEXT NOT NULL,
                previous_decision TEXT,
                reviewer TEXT,
                note TEXT,
                decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_reviews_queue "
                    "ON match_reviews(decision, uncertainty)")
        con.commit()


def import_matches(matches, db_path=DB_PATH):
    """Insert new candidate pairs; refresh scores of still-pending ones.

    Existing human decisions are never overwritten.
    """
    if not isinstance(matches, pd.DataFrame):
        matches = pd.read_csv(matches)
    rows = [(int(r.record_id_A), int(r.record_id_B), str(r.company_A),
             str(r.company_B), float(r.match_score), str(r.match_category))
            for r in matches.itertuples(index=False)]
    with closing(connect(db_path)) as con:
        before = con.execute("SELECT COUNT(*) FROM match_reviews").fetchone()[0]
        con.executemany("""
            INSERT INTO match_reviews
                (record_id_A, record_id_B, company_A, company_B, match_score, match_category)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id_A, record_id_B) DO UPDATE SET
                match_score = excluded.match_score,
                match_category = excluded.match_category
            WHERE match_reviews.decision = 'PENDING'
        """, rows)
        con.commit()
        after = con.execute("SELECT COUNT(*) FROM match_reviews").fetchone()[0]
    return after - before


def set_uncertainty(predictions, db_path=DB_PATH):
    """Store model uncertainty (0..1, 1 = most uncertain) for pending pairs."""
    rows = [(float(u), int(a), int(b)) for a, b, u in
            zip(predictions["record_id_A"], predictions["record_id_B"],
                predictions["uncertainty"])]
    with closing(connect(db_path)) as con:
        con.executemany("UPDATE match_reviews SET uncertainty = ? "
                        "WHERE record_id_A = ? AND record_id_B = ?", rows)
        con.commit()


def get_next_pair(strategy="uncertain", skip_ids=(), db_path=DB_PATH):
    """Next pending pair as a Series, or None.

    strategy: 'uncertain' (active learning), 'score' (highest score first)
    or 'random'.
    """
    order = {
        "uncertain": "uncertainty DESC, match_score DESC",
        "score": "match_score DESC",
        "random": "RANDOM()",
    }[strategy]
    skip_ids = [int(i) for i in skip_ids]
    clause = f"AND id NOT IN ({','.join('?' * len(skip_ids))})" if skip_ids else ""
    query = (f"SELECT * FROM match_reviews WHERE decision = 'PENDING' {clause} "
             f"ORDER BY {order} LIMIT 1")
    with closing(connect(db_path)) as con:
        df = pd.read_sql_query(query, con, params=skip_ids)
    return None if df.empty else df.iloc[0]


def save_decision(review_id, decision, reviewer="anonymous", note="", db_path=DB_PATH):
    """Record a decision and append it to the audit history."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid review decision: {decision}")
    with closing(connect(db_path)) as con:
        row = con.execute("SELECT decision FROM match_reviews WHERE id = ?",
                          (int(review_id),)).fetchone()
        if row is None:
            raise KeyError(f"Review id {review_id} not found")
        con.execute("""UPDATE match_reviews
                       SET decision = ?, reviewed_at = CURRENT_TIMESTAMP,
                           reviewer = ?, review_note = ?
                       WHERE id = ?""", (decision, reviewer, note, int(review_id)))
        con.execute("""INSERT INTO review_history
                       (review_id, decision, previous_decision, reviewer, note)
                       VALUES (?, ?, ?, ?, ?)""",
                    (int(review_id), decision, row[0], reviewer, note))
        con.commit()


def undo_last_decision(reviewer="anonymous", db_path=DB_PATH):
    """Reset the most recent APPROVED/REJECTED decision back to PENDING."""
    with closing(connect(db_path)) as con:
        row = con.execute("""SELECT id FROM match_reviews
                             WHERE decision IN ('APPROVED','REJECTED')
                             ORDER BY reviewed_at DESC, id DESC LIMIT 1""").fetchone()
    if row is None:
        return None
    save_decision(row[0], "PENDING", reviewer, "undo", db_path)
    return row[0]


def get_review_statistics(db_path=DB_PATH):
    with closing(connect(db_path)) as con:
        return pd.read_sql_query(
            "SELECT decision, COUNT(*) AS total FROM match_reviews GROUP BY decision", con)


def get_decision_counts(db_path=DB_PATH):
    stats = get_review_statistics(db_path)
    counts = dict(zip(stats["decision"], stats["total"]))
    return {k: int(counts.get(k, 0)) for k in ("PENDING", "APPROVED", "REJECTED")}


def get_review_history(limit=200, db_path=DB_PATH):
    with closing(connect(db_path)) as con:
        return pd.read_sql_query("""
            SELECT h.decided_at, h.review_id, m.company_A, m.company_B,
                   h.previous_decision, h.decision, h.reviewer, h.note
            FROM review_history h JOIN match_reviews m ON m.id = h.review_id
            ORDER BY h.id DESC LIMIT ?""", con, params=(limit,))


def get_reviewed_pairs(db_path=DB_PATH):
    with closing(connect(db_path)) as con:
        return pd.read_sql_query(
            "SELECT * FROM match_reviews WHERE decision IN ('APPROVED','REJECTED')", con)


if __name__ == "__main__":
    initialize_database()
    if MATCHES_PATH.exists():
        added = import_matches(MATCHES_PATH)
        print(f"Imported candidate pairs ({added:,} new).")
    print("\n========== DATATWIN REVIEW STATISTICS ==========")
    print(get_review_statistics().to_string(index=False))
