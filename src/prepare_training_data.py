"""Turn human review decisions into ML labels (APPROVED=1, REJECTED=0)."""
import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR, TRAINING_PATH
from src.review_manager import get_reviewed_pairs


def prepare_training_data(db_path=DB_PATH, output_path=TRAINING_PATH):
    df = get_reviewed_pairs(db_path)
    if df.empty:
        print("No reviewed matches available yet. Approve or reject some pairs first.")
        return None
    df["label"] = df["decision"].map({"APPROVED": 1, "REJECTED": 0})
    training = df[["record_id_A", "record_id_B", "company_A", "company_B",
                   "match_score", "match_category", "label", "reviewed_at"]].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    training.to_csv(output_path, index=False)

    print("\n========== TRAINING DATA REPORT ==========")
    print("Total reviewed pairs:", len(training))
    print(training["label"].value_counts().rename({1: "approved", 0: "rejected"}).to_string())
    print("Saved:", output_path)
    return training


if __name__ == "__main__":
    prepare_training_data()
