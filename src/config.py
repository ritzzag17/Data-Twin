"""Central configuration for DataTwin: paths, weights and thresholds."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "business_records.csv"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "datatwin.db"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

# Report files
MATCHES_PATH = REPORT_DIR / "potential_matches.csv"
BLOCKING_REPORT_PATH = REPORT_DIR / "blocking_report.csv"
EXPLANATIONS_PATH = REPORT_DIR / "match_explanations.csv"
JUDGE_PATH = REPORT_DIR / "ai_judge_results.csv"
JUDGE_EVAL_PATH = REPORT_DIR / "judge_evaluation.json"
PROMPT_PATH = REPORT_DIR / "prompt_templates.txt"
PREDICTIONS_PATH = REPORT_DIR / "ml_predictions.csv"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
MISSING_REPORT_PATH = REPORT_DIR / "missing_values_report.csv"
QUALITY_SUMMARY_PATH = REPORT_DIR / "data_quality_summary.json"
TRUST_PATH = REPORT_DIR / "data_trust_scores.csv"
REPAIR_PATH = REPORT_DIR / "data_repair_suggestions.csv"
CLUSTERS_PATH = REPORT_DIR / "entity_clusters.csv"
GOLDEN_PATH = REPORT_DIR / "golden_records.csv"
CLUSTER_SUMMARY_PATH = REPORT_DIR / "cluster_summary.json"
TRAINING_PATH = PROCESSED_DIR / "training_data.csv"

# Columns the pipeline understands (canonical names)
CORE_COLUMNS = [
    "company_name", "website_domain", "main_country", "main_city",
    "main_address_raw_text", "primary_phone", "primary_email",
]
OPTIONAL_COLUMNS = [
    "main_street_number", "main_latitude", "main_longitude",
    "naics_2022_primary_code", "year_founded",
]

# ---- Blocking ----
MAX_BLOCK_SIZE = 100          # blocks larger than this are sub-blocked
MAX_CANDIDATE_PAIRS = 3_000_000

# ---- Scoring ----
MIN_MATCH_SCORE = 0.60        # pairs below this are not stored
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.75

SCORE_WEIGHTS = {
    "name": 0.50,
    "website": 0.18,
    "phone": 0.08,
    "email": 0.06,
    "address": 0.08,
    "city": 0.05,
    "country": 0.05,
}
COUNTRY_CONFLICT_PENALTY = 0.20
CITY_CONFLICT_PENALTY = 0.10
GEO_CLOSE_KM = 0.15

# ---- ML ----
FEATURES = [
    "name_similarity", "name_token_set", "name_jaro_winkler",
    "address_similarity",
    "same_website", "same_phone", "same_email",
    "same_city", "same_country", "same_street_number",
    "geo_close", "same_industry",
    "country_conflict", "city_conflict",
]
MIN_RECOMMENDED_LABELS = 300

# ---- Trust score ----
FIELD_WEIGHTS = {
    "company_name": 25, "website_domain": 20, "main_country": 15,
    "main_city": 10, "main_address_raw_text": 15,
    "primary_phone": 10, "primary_email": 5,
}
TRUST_COMPONENT_WEIGHTS = {"completeness": 0.60, "validity": 0.25, "uniqueness": 0.15}
HIGH_QUALITY = 85
MEDIUM_QUALITY = 60
SPARSE_COLUMN_THRESHOLD = 95.0   # % missing above which a column is flagged

# ---- Clustering ----
CLUSTER_PROB_THRESHOLD = 0.90
CLUSTER_MAX_SIZE = 25

# ---- LLM judge (optional) ----
LLM_MODEL = "claude-sonnet-5-5"   # override with DATATWIN_LLM_MODEL
