"""DataTwin dashboard:  streamlit run app/app.py"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import config as C  # noqa: E402
from src import review_manager as rm  # noqa: E402
from src.clustering import GOLDEN_FIELDS, build_cluster_tables, pairs_from_humans  # noqa: E402
from src.data_quality import profile  # noqa: E402
from src.duplicate_detection import find_candidates  # noqa: E402
from src.match_explainer import explain_pair  # noqa: E402
from src.prompt_engine import local_judge  # noqa: E402
from src.trust_score import compute_trust  # noqa: E402

st.set_page_config(page_title="DataTwin", page_icon="✦", layout="wide")

PAGES = ["Overview", "Upload & Analyze", "Data Quality", "Entity Resolution",
         "ML Predictions", "Human Review", "Data Trust", "Clusters", "Reports", "About"]
DISPLAY_FIELDS = ["company_name", "website_domain", "main_country", "main_city",
                  "main_address_raw_text", "primary_phone", "primary_email",
                  "year_founded"]
MAX_UPLOAD_ROWS = 20_000
RUN_HINT = "Run `python run_pipeline.py` from the project root to generate this report."


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner=False)
def load_report(path, mtime):
    """Read a CSV report (mtime is part of the cache key so it refreshes)."""
    return pd.read_csv(path, low_memory=False)


def report(path):
    path = Path(path)
    if not path.exists():
        return None
    return load_report(str(path), path.stat().st_mtime)


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner="Loading dataset...")
def load_raw():
    if not C.RAW_PATH.exists():
        return None
    header = pd.read_csv(C.RAW_PATH, nrows=0).columns
    wanted = [c for c in dict.fromkeys(DISPLAY_FIELDS + GOLDEN_FIELDS) if c in header]
    return pd.read_csv(C.RAW_PATH, usecols=wanted, low_memory=False)


def fmt(value):
    if pd.isna(value) or str(value).strip() == "":
        return "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def pair_table(raw, a, b):
    rows = [{"Field": f.replace("_", " ").title(), "Business A": fmt(raw.iloc[a].get(f)),
             "Business B": fmt(raw.iloc[b].get(f))} for f in DISPLAY_FIELDS if f in raw.columns]
    return pd.DataFrame(rows)


def need(df, label="report"):
    if df is None:
        st.info(f"The {label} is not available yet. {RUN_HINT}")
        return False
    return True


def download(df, name, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name,
                       mime="text/csv")


def bar(df, x, y, title, height=320):
    fig = px.bar(df, x=x, y=y, title=title, height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ✦ DataTwin")
    st.caption("BUSINESS DATA INTELLIGENCE")
    st.divider()
    page = st.radio("WORKSPACE", PAGES, key="selected_page")
    st.divider()
    st.caption("Human-assisted entity resolution · Prototype")

st.title(page)

# ================================================================= OVERVIEW
if page == "Overview":
    matches = report(C.MATCHES_PATH)
    trust = report(C.TRUST_PATH)
    quality = load_json(C.QUALITY_SUMMARY_PATH)
    metrics = load_json(C.METRICS_PATH)
    counts = rm.get_decision_counts() if C.DB_PATH.exists() else {"PENDING": 0, "APPROVED": 0, "REJECTED": 0}

    c = st.columns(5)
    c[0].metric("Records", f"{quality['records']:,}" if quality else "—")
    c[1].metric("Potential matches", f"{len(matches):,}" if matches is not None else "—")
    c[2].metric("High confidence",
                f"{(matches['match_category'] == 'High Confidence').sum():,}" if matches is not None else "—")
    c[3].metric("Avg trust score", f"{trust['trust_score'].mean():.1f}" if trust is not None else "—")
    c[4].metric("Human-reviewed", counts["APPROVED"] + counts["REJECTED"])

    left, right = st.columns(2)
    if matches is not None:
        cats = matches["match_category"].value_counts().reset_index()
        cats.columns = ["Category", "Pairs"]
        with left:
            bar(cats, "Category", "Pairs", "Candidate pairs by confidence")
    if trust is not None:
        q = trust["quality_category"].value_counts().reset_index()
        q.columns = ["Category", "Records"]
        with right:
            fig = px.pie(q, names="Category", values="Records", title="Record quality", hole=0.5, height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Next best actions")
    labels = counts["APPROVED"] + counts["REJECTED"]
    if labels < C.MIN_RECOMMENDED_LABELS:
        st.warning(f"Only {labels} pairs are human-labelled (recommended: {C.MIN_RECOMMENDED_LABELS}+). "
                   "Open **Human Review** - it shows the pairs the models are least sure about first.")
    else:
        st.success(f"{labels} labelled pairs - enough for a meaningful model evaluation.")
    if metrics and metrics.get("cross_validation"):
        cv = metrics["cross_validation"]
        best = max(cv, key=lambda k: cv[k]["f1"]["mean"])
        st.info(f"Best model in cross-validation: **{best}** (F1 {cv[best]['f1']['mean']:.2f}).")
    if matches is None:
        st.info(RUN_HINT)

# ============================================================ UPLOAD/ANALYZE
elif page == "Upload & Analyze":
    st.write(f"Upload a CSV of business records (up to {MAX_UPLOAD_ROWS:,} rows) to profile quality, "
             "find duplicates and score trust. Your file is processed in memory only.")
    upload = st.file_uploader("Business records (CSV)", type=["csv"])
    ALIASES = {
        "company_name": ["company_name", "name", "business_name", "company", "organization"],
        "website_domain": ["website_domain", "website", "domain", "url", "web"],
        "main_country": ["main_country", "country"],
        "main_city": ["main_city", "city", "town"],
        "main_address_raw_text": ["main_address_raw_text", "address", "street_address", "full_address"],
        "primary_phone": ["primary_phone", "phone", "telephone", "mobile"],
        "primary_email": ["primary_email", "email", "e_mail"],
    }
    if upload is not None:
        df = pd.read_csv(upload, low_memory=False)
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
        if len(df) > MAX_UPLOAD_ROWS:
            st.warning(f"Only the first {MAX_UPLOAD_ROWS:,} rows will be analysed.")
            df = df.head(MAX_UPLOAD_ROWS)

        st.subheader("Map your columns")
        lowered = {c.lower().strip(): c for c in df.columns}
        mapping = {}
        cols = st.columns(4)
        for i, (canon, aliases) in enumerate(ALIASES.items()):
            guess = next((lowered[a] for a in aliases if a in lowered), None)
            options = ["(none)"] + list(df.columns)
            mapping[canon] = cols[i % 4].selectbox(
                canon, options, index=options.index(guess) if guess else 0, key=f"map_{canon}")

        if st.button("Run analysis", type="primary"):
            if mapping["company_name"] == "(none)":
                st.error("A company name column is required.")
            else:
                renamed = df.rename(columns={v: k for k, v in mapping.items() if v != "(none)"})
                with st.spinner("Profiling, matching and scoring..."):
                    summary, missing = profile(renamed)
                    matches, stats = find_candidates(renamed)
                    scores, issues = compute_trust(renamed)
                st.session_state["upload_result"] = (summary, missing, matches, scores, issues)

    if "upload_result" in st.session_state:
        summary, missing, matches, scores, issues = st.session_state["upload_result"]
        c = st.columns(4)
        c[0].metric("Records", f"{summary['records']:,}")
        c[1].metric("Exact duplicate rows", summary["exact_duplicate_rows"])
        c[2].metric("Potential matches", f"{len(matches):,}")
        c[3].metric("Avg trust score", f"{scores['trust_score'].mean():.1f}")
        t1, t2, t3 = st.tabs(["Potential duplicates", "Trust scores", "Repair suggestions"])
        with t1:
            st.dataframe(matches.head(500))
            if len(matches):
                download(matches, "uploaded_matches.csv")
        with t2:
            st.dataframe(scores.head(500))
            download(scores, "uploaded_trust_scores.csv")
        with t3:
            st.dataframe(issues.head(500))
            download(issues, "uploaded_repair_suggestions.csv")

# =============================================================== DATA QUALITY
elif page == "Data Quality":
    missing = report(C.MISSING_REPORT_PATH)
    quality = load_json(C.QUALITY_SUMMARY_PATH)
    if need(missing, "data quality report"):
        if quality:
            c = st.columns(4)
            c[0].metric("Records", f"{quality['records']:,}")
            c[1].metric("Exact duplicate rows", quality["exact_duplicate_rows"])
            c[2].metric("Missing company names", quality.get("missing_company_names", "—"))
            c[3].metric("Sparse columns", len(quality["sparse_columns"]))
        top = st.slider("Columns to show", 5, 40, 15)
        bar(missing.head(top), "Missing_Percentage", "Column", "Missing values by column (%)", height=max(320, top * 22))
        if quality and quality["sparse_columns"]:
            st.warning(f"These columns are almost empty (>= {C.SPARSE_COLUMN_THRESHOLD:.0f}% missing) and are "
                       "good candidates to drop: " + ", ".join(quality["sparse_columns"]))
        if quality:
            st.subheader("Format validation")
            fmt_df = pd.DataFrame(quality["invalid_formats"]).T.reset_index()
            fmt_df.columns = ["Field", "Present", "Invalid"]
            st.dataframe(fmt_df)
        st.dataframe(missing)

# ========================================================= ENTITY RESOLUTION
elif page == "Entity Resolution":
    matches = report(C.MATCHES_PATH)
    raw = load_raw()
    if need(matches, "potential matches report"):
        c1, c2, c3 = st.columns([2, 2, 3])
        cats = c1.multiselect("Category", ["High Confidence", "Medium Confidence", "Low Confidence"],
                              default=["High Confidence", "Medium Confidence"])
        min_score = c2.slider("Minimum score", 0.6, 1.0, 0.75, 0.01)
        search = c3.text_input("Search company name")
        view = matches[matches["match_category"].isin(cats) & (matches["match_score"] >= min_score)]
        if search:
            mask = (view["company_A"].astype(str).str.contains(search, case=False, na=False)
                    | view["company_B"].astype(str).str.contains(search, case=False, na=False))
            view = view[mask]
        st.caption(f"{len(view):,} pairs")
        show_cols = ["record_id_A", "record_id_B", "company_A", "company_B", "city_A", "city_B",
                     "match_score", "match_category"]
        st.dataframe(view[show_cols].head(1000))

        st.subheader("Inspect a pair")
        if len(view):
            idx = st.number_input("Row number in the table above", 0, min(len(view), 1000) - 1, 0)
            row = view.iloc[int(idx)]
            if raw is not None:
                st.table(pair_table(raw, int(row["record_id_A"]), int(row["record_id_B"])))
            info = explain_pair(row.to_dict())
            verdict = local_judge(row.to_dict())
            st.write(f"**Score:** {row['match_score']} · **Rule-based judge:** {verdict.decision} "
                     f"({verdict.confidence:.0%})")
            st.write(info["explanation"])
            if info["conflicting_evidence"]:
                st.warning("Conflicting evidence: " + info["conflicting_evidence"])
        blocking = report(C.BLOCKING_REPORT_PATH)
        if blocking is not None:
            with st.expander("Blocking report (how candidate pairs were generated)"):
                st.dataframe(blocking)

# ============================================================== ML PREDICTIONS
elif page == "ML Predictions":
    metrics = load_json(C.METRICS_PATH)
    preds = report(C.PREDICTIONS_PATH)
    if metrics is None:
        st.info("No trained models yet. Review some pairs in **Human Review**, then run "
                "`python run_pipeline.py --only training_data,train,predict`.")
    else:
        c = st.columns(3)
        c[0].metric("Labelled pairs", metrics["n_labels"])
        c[1].metric("Approved", metrics["n_approved"])
        c[2].metric("Rejected", metrics["n_rejected"])
        for w in metrics.get("warnings", []):
            st.warning(w)
        cv = metrics.get("cross_validation")
        if cv:
            st.subheader(f"Cross-validated performance ({metrics['cv_folds']}-fold, repeated)")
            rows = [{"Model": name, **{m: f"{v[m]['mean']:.2f} ± {v[m]['std']:.2f}"
                                       for m in ("precision", "recall", "f1", "roc_auc")}}
                    for name, v in cv.items()]
            st.dataframe(pd.DataFrame(rows))
            st.caption("`rule_baseline` predicts MATCH when the rule score is >= 0.90. A model is only "
                       "useful if it beats this baseline on held-out data.")
        fi = pd.DataFrame(metrics["feature_importance"].items(), columns=["Feature", "Importance"])
        bar(fi.sort_values("Importance"), "Importance", "Feature", "Random Forest feature importance", 380)
    if preds is not None:
        st.subheader("Predictions")
        only_disagree = st.checkbox("Only pairs where the models disagree")
        view = preds[~preds["model_agreement"]] if only_disagree else preds
        st.dataframe(view.sort_values("uncertainty", ascending=False).head(1000))
        st.caption("Predictions are experimental suggestions - not verified matches.")

# =============================================================== HUMAN REVIEW
elif page == "Human Review":
    if not C.DB_PATH.exists():
        st.info("Review database not found. " + RUN_HINT)
    else:
        rm.initialize_database()
        raw = load_raw()
        counts = rm.get_decision_counts()
        c = st.columns(3)
        c[0].metric("Pending", f"{counts['PENDING']:,}")
        c[1].metric("Approved", counts["APPROVED"])
        c[2].metric("Rejected", counts["REJECTED"])

        s1, s2 = st.columns(2)
        reviewer = s1.text_input("Reviewer name", "anonymous")
        strategy = s2.selectbox("Queue order", ["uncertain", "score", "random"],
                                help="'uncertain' (active learning) shows the pairs the models are least "
                                     "sure about first - the most valuable labels for training.")
        st.session_state.setdefault("skipped", [])
        pair = rm.get_next_pair(strategy, st.session_state["skipped"])

        if pair is None:
            st.success("No more pending pairs.")
        elif raw is None:
            st.error("Raw dataset not found at data/raw/business_records.csv")
        else:
            a, b = int(pair["record_id_A"]), int(pair["record_id_B"])
            st.write(f"**Review ID {pair['id']}** · score {pair['match_score']} · {pair['match_category']}")
            st.table(pair_table(raw, a, b))

            preds = report(C.PREDICTIONS_PATH)
            if preds is not None:
                hit = preds[(preds["record_id_A"] == a) & (preds["record_id_B"] == b)]
                if not hit.empty:
                    r = hit.iloc[0]
                    st.info(f"ML suggestion: **{'Likely match' if r['predicted_label'] == 1 else 'Likely non-match'}** "
                            f"(ensemble probability {r['ensemble_probability']:.0%}, models "
                            f"{'agree' if r['model_agreement'] else 'disagree'}).")
            matches = report(C.MATCHES_PATH)
            if matches is not None:
                hit = matches[(matches["record_id_A"] == a) & (matches["record_id_B"] == b)]
                if not hit.empty:
                    info = explain_pair(hit.iloc[0].to_dict())
                    st.write(info["explanation"])

            note = st.text_input("Optional note", key=f"note_{pair['id']}")
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("✅ Approve match"):
                rm.save_decision(int(pair["id"]), "APPROVED", reviewer, note)
                st.rerun()
            if b2.button("❌ Reject match"):
                rm.save_decision(int(pair["id"]), "REJECTED", reviewer, note)
                st.rerun()
            if b3.button("⏭️ Skip"):
                st.session_state["skipped"].append(int(pair["id"]))
                st.rerun()
            if b4.button("↩️ Undo last decision"):
                rm.undo_last_decision(reviewer)
                st.rerun()

        with st.expander("Audit history (latest decisions)"):
            st.dataframe(rm.get_review_history(100))

# ================================================================= DATA TRUST
elif page == "Data Trust":
    scores = report(C.TRUST_PATH)
    issues = report(C.REPAIR_PATH)
    if need(scores, "trust report"):
        c = st.columns(4)
        c[0].metric("Average trust", f"{scores['trust_score'].mean():.1f}")
        for col, cat in zip(c[1:], ["High Quality", "Medium Quality", "Low Quality"]):
            col.metric(cat, f"{(scores['quality_category'] == cat).sum():,}")
        fig = px.histogram(scores, x="trust_score", nbins=40, title="Trust score distribution", height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Trust = 60% completeness + 25% format validity + 15% uniqueness. It measures how complete "
                   "and well-formed a record is - not whether the facts are correct.")
        if issues is not None:
            counts = issues["issue"].value_counts().reset_index()
            counts.columns = ["Issue", "Count"]
            bar(counts, "Issue", "Count", "Issues by type")
            kind = st.selectbox("Filter issues", ["All"] + counts["Issue"].tolist())
            view = issues if kind == "All" else issues[issues["issue"] == kind]
            st.dataframe(view.head(1000))
        st.subheader("Lowest-trust records")
        st.dataframe(scores.sort_values("trust_score").head(200))

# ================================================================== CLUSTERS
elif page == "Clusters":
    st.write("If A = B and B = C, then A, B and C are one business. Clusters group such records and "
             "merge them into a single **golden record**.")
    raw = load_raw()
    source = st.radio("Build clusters from", ["Human-approved pairs (safe)",
                                              "High-confidence ML predictions"], horizontal=True)
    if raw is None:
        st.error("Raw dataset not found.")
    else:
        pairs = []
        if source.startswith("Human"):
            pairs = pairs_from_humans() if C.DB_PATH.exists() else []
        else:
            preds = report(C.PREDICTIONS_PATH)
            thr = st.slider("Minimum ensemble probability", 0.5, 1.0, C.CLUSTER_PROB_THRESHOLD, 0.01)
            if need(preds, "predictions"):
                strong = preds[(preds["ensemble_probability"] >= thr) & (preds["match_score"] >= 0.90)]
                pairs = list(zip(strong["record_id_A"], strong["record_id_B"]))
        members, golden, summary = build_cluster_tables(raw, pairs, source)
        if members.empty:
            st.info("No clusters yet - approve some pairs in Human Review or lower the probability threshold.")
        else:
            c = st.columns(4)
            c[0].metric("Clusters", summary["clusters"])
            c[1].metric("Records in clusters", summary["records_in_clusters"])
            c[2].metric("Largest cluster", summary["largest_cluster"])
            c[3].metric("Duplicates removed if merged", summary["duplicates_removed_if_merged"])
            if summary["clusters_needing_review"]:
                st.warning(f"{summary['clusters_needing_review']} clusters are unusually large and may contain "
                           "false links (chaining). Review them manually.")
            st.subheader("Golden records")
            st.dataframe(golden)
            download(golden, "golden_records.csv", "Download golden records")
            pick = st.selectbox("Inspect cluster", golden["cluster_id"].tolist())
            ids = members[members["cluster_id"] == pick]["record_id"].tolist()
            st.write("**Member records**")
            st.dataframe(raw.loc[ids, [c for c in DISPLAY_FIELDS if c in raw.columns]])

# ================================================================== REPORTS
elif page == "Reports":
    files = sorted(p for p in C.REPORT_DIR.glob("*") if p.suffix in {".csv", ".json", ".txt"})
    if not files:
        st.info(RUN_HINT)
    for p in files:
        c1, c2, c3 = st.columns([4, 1, 1])
        c1.write(f"**{p.name}**")
        c2.caption(f"{p.stat().st_size / 1e6:.1f} MB")
        c3.download_button("Download", p.read_bytes(), file_name=p.name, key=f"dl_{p.name}")

# =================================================================== ABOUT
else:
    st.markdown("""
**DataTwin** is an explainable, human-assisted entity-resolution and data-quality prototype for business records.

**Pipeline:** profile → multi-key blocking → weighted scoring with conflict penalties → explanations →
rule-based (optionally LLM) judge → human review with active learning → ML models with cross-validation →
clusters & golden records.

**Limitations**
- Candidate matches are suggestions, not verified duplicates.
- ML models are only as good as the number of human labels; check *ML Predictions* for cross-validated metrics.
- The trust score measures completeness, format validity and uniqueness - not factual accuracy.
- The LLM judge is optional and only sees the supplied evidence.
- Human verification is required before accepting matching decisions.
""")
