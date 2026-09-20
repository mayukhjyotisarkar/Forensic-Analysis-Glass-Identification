"""Streamlit demo: enter refractive index + oxide %, get the predicted glass type.

Run with:   streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, FEATURES
from src.data_loader import load_data
from src.predict import load_metadata, load_model, out_of_range_warnings, predict

st.set_page_config(page_title="Forensic Glass Identification", page_icon="🔎")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data():
    return load_data()


try:
    model, meta, data = get_model(), load_metadata(), get_data()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

st.title("🔎 Forensic Glass Identification")
st.caption(f"Model: **{meta['model_name']}** · trained on {meta['n_rows']} glass samples (UCI dataset). "
           "Educational demo, not evidence-grade forensic software.")

# --- initialise widget state with typical (median) values --------------------
for f in FEATURES:
    st.session_state.setdefault(f, float(data[f].median()))

# --- sidebar: load a real example -------------------------------------------
with st.sidebar:
    st.header("Try a real sample")
    cls = st.selectbox("Class", list(CLASS_NAMES), format_func=lambda c: f"{c}: {CLASS_NAMES[c]}")
    if st.button("Load random example"):
        row = data[data["Type"] == cls].sample(1).iloc[0]
        for f in FEATURES:
            st.session_state[f] = float(row[f])
        st.rerun()

# --- inputs -------------------------------------------------------------------
st.subheader("Measurements")
cols = st.columns(3)
values = {}
for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        values[f] = st.number_input(
            f"{f}" + ("" if f == "RI" else " (wt %)"),
            step=0.0001 if f == "RI" else 0.01,
            format="%.5f" if f == "RI" else "%.2f",
            key=f,
        )

# --- prediction ---------------------------------------------------------------
if st.button("Identify glass type", type="primary"):
    result = predict(model, values).iloc[0]
    st.success(f"**Type {int(result['predicted_type'])}: {result['description']}**")
    if "confidence" in result:
        st.metric("Model confidence", f"{result['confidence']:.0%}")
        proba = {f"{c}: {CLASS_NAMES[c]}": float(result[f"P(type {c})"]) for c in model.classes_}
        st.bar_chart(pd.Series(proba, name="probability"), horizontal=True)
        if result["confidence"] < 0.6:
            st.info("Low confidence: types 1 and 2 (float vs non-float building glass) "
                    "overlap heavily in this dataset, so treat this as a weak indication.")
    for msg in out_of_range_warnings(pd.DataFrame([values]), meta["feature_ranges"]):
        st.warning(msg.replace("Row 0: ", ""))
