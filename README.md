# DataTwin — AI-Powered Business Data Quality & Entity Resolution

DataTwin finds duplicate business records, explains *why* two records match, lets a human
confirm or reject each pair, learns from those decisions, and finally merges confirmed
duplicates into one clean **golden record**.

```
raw data → profiling → blocking → scoring → explanations → judge (rules / optional LLM)
        → human review (active learning) → ML models (cross-validated) → clusters & golden records
                                   └────────────→ Data Trust Score
```

## What's new in this version

| Area | Before | Now |
|---|---|---|
| Blocking | one 5-letter name key; blocks > 100 silently dropped | 6 keys (domain, phone, email, name prefix, first token, sorted tokens); oversized blocks are **sub-blocked** and every skipped record is reported (`reports/blocking_report.csv`) |
| Features | 5 (name, address, website, city, country) | 14: + phone, email, street number, GPS distance, industry, token-set / Jaro-Winkler name similarity, **country/city conflict** flags; legal suffixes (GmbH, Ltd, …) and accents are normalised |
| Scoring | name-heavy, no penalties | weighted evidence with penalties for conflicting location |
| ML | fit on all labels, no evaluation | **repeated stratified cross-validation**, precision / recall / F1 / ROC-AUC, compared against a rule baseline (`reports/model_metrics.json`) |
| Review | random pair, decision overwritten | **active-learning queue** (most uncertain pair first), reviewer name, notes, undo, full audit history |
| Judge | rules on 100 pairs | rules on top pairs + all reviewed pairs, **optional LLM** (Anthropic API) only for UNCERTAIN pairs, evidence verification, agreement with human labels |
| Trust score | completeness only | completeness (60%) + format validity (25%) + uniqueness (15%) |
| Output | list of pairs | **entity clusters + golden records** |
| Code | duplicated logic, relative paths, row loops | shared `src/` modules, `config.py`, `run_pipeline.py`, vectorised trust score, unit tests |

## Project structure

```text
DataTwin/
├── app/app.py              # Streamlit dashboard (10 pages)
├── data/
│   ├── raw/                # put business_records.csv here
│   ├── processed/          # training data
│   └── datatwin.db         # review database (SQLite)
├── models/                 # trained models (generated)
├── notebooks/01_eda.ipynb
├── reports/                # generated CSV / JSON reports
├── src/
│   ├── config.py           # paths, weights, thresholds - tune things here
│   ├── text_utils.py       # normalisation + similarity (RapidFuzz, pure-Python fallback)
│   ├── features.py         # pair features + match score
│   ├── blocking.py         # multi-key blocking
│   ├── data_quality.py     # dataset profiling
│   ├── duplicate_detection.py
│   ├── match_explainer.py  # supporting vs conflicting evidence
│   ├── prompt_engine.py    # rule judge + optional LLM judge
│   ├── review_manager.py   # SQLite review store, audit trail, active learning
│   ├── prepare_training_data.py
│   ├── train_model.py      # training + cross-validation
│   ├── predict_matches.py
│   ├── clustering.py       # clusters + golden records
│   ├── trust_score.py
│   └── review_interface.py # stand-alone review screen
├── tests/test_core.py
├── run_pipeline.py
├── requirements.txt / requirements-dev.txt
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Using your existing data:** copy `data/raw/business_records.csv` and (optionally)
`data/datatwin.db` from the old project. The old database is migrated in place — your
existing approved/rejected decisions are kept and used as training labels.

## Running

```bash
python run_pipeline.py                 # everything, in the right order
python run_pipeline.py --only matching,train,predict
python run_pipeline.py --skip trust,clusters
streamlit run app/app.py               # dashboard
```

Individual steps can also be run from the project root, e.g. `python -m src.train_model`.

| Step | Command | Output |
|---|---|---|
| quality | `python -m src.data_quality` | `missing_values_report.csv`, `data_quality_summary.json` |
| matching | `python -m src.duplicate_detection` | `potential_matches.csv`, `blocking_report.csv` |
| review_db | (pipeline) | imports pairs into `data/datatwin.db` |
| training_data | `python -m src.prepare_training_data` | `data/processed/training_data.csv` |
| train | `python -m src.train_model` | `models/*.joblib`, `model_metrics.json` |
| predict | `python -m src.predict_matches` | `ml_predictions.csv`, updates review queue |
| explain | `python -m src.match_explainer` | `match_explanations.csv` |
| judge | `python -m src.prompt_engine` | `ai_judge_results.csv`, `judge_evaluation.json` |
| trust | `python -m src.trust_score` | `data_trust_scores.csv`, `data_repair_suggestions.csv` |
| clusters | `python -m src.clustering [humans\|predictions]` | `entity_clusters.csv`, `golden_records.csv` |

### Recommended workflow

1. `python run_pipeline.py` — first full run.
2. Open **Human Review** and label pairs. The queue shows the pairs the models are *least sure* about first.
3. After every ~50 labels: `python run_pipeline.py --only training_data,train,predict`.
4. Watch the cross-validated F1 on **ML Predictions**; aim for 300+ labels.
5. Build clusters / golden records on the **Clusters** page.

### Optional: LLM judge

```bash
export ANTHROPIC_API_KEY=...            # optional: DATATWIN_LLM_MODEL=<model name>
python run_pipeline.py --only judge --llm
```
Only pairs the rules mark UNCERTAIN are sent to the model (max 50 per run), and its answer is
validated, evidence-checked and compared with your human labels. Without a key everything
runs locally.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Limitations

- Candidate matches are suggestions, not verified duplicates.
- With few labels the ML models are unreliable — see `model_metrics.json` (a warning is shown below 300 labels). Model agreement is **not** accuracy.
- The Data Trust Score measures completeness, format validity and uniqueness, not factual accuracy.
- Clusters built from ML predictions can chain false links; large clusters are flagged for review.
- Same name + same website does not always mean the same entity (chains, franchises, branches) — that is exactly what the human review is for.

## Future work

Learning score weights from labels, phonetic/transliteration matching for non-Latin names,
embedding-based blocking, a REST API, and a Dockerfile.
