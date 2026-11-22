"""Stand-alone human review screen:  streamlit run src/review_interface.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import review_manager as rm  # noqa: E402
from src.config import RAW_PATH  # noqa: E402

FIELDS = ["company_name", "website_domain", "main_country", "main_city",
          "main_address_raw_text", "primary_phone", "primary_email"]

st.set_page_config(page_title="DataTwin | Human Review", page_icon="🔍", layout="wide")


@st.cache_data
def load_dataset():
    header = pd.read_csv(RAW_PATH, nrows=0).columns
    return pd.read_csv(RAW_PATH, usecols=[c for c in FIELDS if c in header], low_memory=False)


def show(record, title):
    st.subheader(title)
    for f in FIELDS:
        value = record.get(f)
        if pd.notna(value) and str(value).strip():
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            st.write(f"**{f.replace('_', ' ').title()}:** {value}")
        else:
            st.write(f"**{f.replace('_', ' ').title()}:** Not available")


st.title("DataTwin — Human Review Center")
rm.initialize_database()
counts = rm.get_decision_counts()
c1, c2, c3 = st.columns(3)
c1.metric("Pending", counts["PENDING"])
c2.metric("Approved", counts["APPROVED"])
c3.metric("Rejected", counts["REJECTED"])

reviewer = st.sidebar.text_input("Reviewer name", "anonymous")
strategy = st.sidebar.selectbox("Queue order", ["uncertain", "score", "random"],
                                help="'uncertain' shows pairs the ML models are least sure about first.")
st.session_state.setdefault("skipped", [])

pair = rm.get_next_pair(strategy, st.session_state["skipped"])
if pair is None:
    st.success("No more pending pairs.")
    st.stop()

data = load_dataset()
st.write(f"**Review ID:** {pair['id']} · **Score:** {pair['match_score']} · **{pair['match_category']}**")
left, right = st.columns(2)
with left:
    show(data.iloc[int(pair["record_id_A"])], "Business A")
with right:
    show(data.iloc[int(pair["record_id_B"])], "Business B")

note = st.text_input("Optional note")
a, b, c = st.columns(3)
if a.button("✅ Approve match"):
    rm.save_decision(int(pair["id"]), "APPROVED", reviewer, note)
    st.rerun()
if b.button("❌ Reject match"):
    rm.save_decision(int(pair["id"]), "REJECTED", reviewer, note)
    st.rerun()
if c.button("⏭️ Skip"):
    st.session_state["skipped"].append(int(pair["id"]))
    st.rerun()
