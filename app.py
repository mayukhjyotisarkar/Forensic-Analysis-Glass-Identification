"""Streamlit demo: enter refractive index + oxide %, get predicted glass type & SHAP explanations.

Run with:   streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, FEATURES
from src.data_loader import load_data
from src.predict import load_metadata, load_model, out_of_range_warnings, predict

st.set_page_config(page_title="Forensic Glass Identification", page_icon="🔎", layout="wide")


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

st.title("🔎 Forensic Glass Identification & Analysis System")
st.caption(f"Model Architecture: **{meta['model_name']}** · Trained on {meta['n_rows']} glass samples (UCI dataset). "
           "Educational forensic demo.")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["🔬 Single Sample Analysis & SHAP", "📁 Batch CSV Processing"])

# --- Sidebar: Sample presets ---
with st.sidebar:
    st.header("Preset Forensic Samples")
    cls = st.selectbox("Select Class", list(CLASS_NAMES), format_func=lambda c: f"{c}: {CLASS_NAMES[c]}")
    if st.button("Load Random Example"):
        row = data[data["Type"] == cls].sample(1).iloc[0]
        for f in FEATURES:
            st.session_state[f] = float(row[f])
        st.rerun()

for f in FEATURES:
    st.session_state.setdefault(f, float(data[f].median()))

# ==============================================================================
# TAB 1: Single Sample Analysis
# ==============================================================================
with tab1:
    st.subheader("1. Input Elemental Oxide & Refractive Measurements")
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

    if st.button("Identify Glass Type", type="primary"):
        df_input = pd.DataFrame([values])
        result = predict(model, df_input).iloc[0]
        pred_type = int(result["predicted_type"])
        
        st.markdown("---")
        st.subheader("2. Classification Results")
        
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.success(f"**Predicted Type {pred_type}**: {result['description']}")
            if "confidence" in result:
                st.metric("Model Confidence", f"{result['confidence']:.1%}")
                if result["confidence"] < 0.6:
                    st.info("⚠️ Low confidence warning: float vs non-float building glass overlap heavily in chemistry.")
        
        with res_col2:
            if "confidence" in result:
                proba = {f"{c}: {CLASS_NAMES[c]}": float(result[f"P(type {c})"]) for c in model.classes_}
                st.write("**Class Probabilities**")
                st.bar_chart(pd.Series(proba, name="Probability"), horizontal=True)

        # Extrapolation warnings
        warnings = out_of_range_warnings(df_input, meta["feature_ranges"])
        if warnings:
            st.markdown("---")
            st.subheader("3. Feature Range Check (Out-of-Distribution Warning)")
            for msg in warnings:
                st.warning(msg.replace("Row 0: ", ""))

        # SHAP Explanation Section
        st.markdown("---")
        st.subheader("4. Forensic SHAP Feature Attribution")
        st.caption("SHAP values quantify how each measured element pushes the prediction toward or away from the target class.")

        try:
            import shap
            
            clf = model.named_steps["clf"]
            X_trans = model[:-1].transform(df_input) if len(model.steps) > 1 else df_input
            
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X_trans)
            
            # Find class index corresponding to predicted_type
            class_idx = list(model.classes_).index(pred_type) if pred_type in model.classes_ else 0
            
            fig, ax = plt.subplots(figsize=(8, 4))
            if len(shap_values.shape) == 3: # Multi-class output
                sv_class = shap_values[0, :, class_idx]
            else:
                sv_class = shap_values[0]
                
            features_disp = list(df_input.columns)
            if hasattr(X_trans, "columns"):
                features_disp = list(X_trans.columns)
            
            values_disp = sv_class.values if hasattr(sv_class, "values") else sv_class
            
            y_pos = np.arange(len(features_disp))
            ax.barh(y_pos, values_disp, color=["#e74c3c" if v < 0 else "#2ecc71" for v in values_disp])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features_disp)
            ax.set_xlabel(f"SHAP Value (Impact on Type {pred_type} prediction)")
            ax.set_title(f"SHAP Feature Impact for Predicted Class: {CLASS_NAMES.get(pred_type, pred_type)}")
            fig.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP visualization note: {e}")

# ==============================================================================
# TAB 2: Batch CSV Processing
# ==============================================================================
with tab2:
    st.subheader("Batch Glass Fragment Identification")
    st.markdown("Upload a CSV file containing columns for the 9 features: `RI, Na, Mg, Al, Si, K, Ca, Ba, Fe`.")
    
    # Template download
    sample_template = pd.DataFrame([
        {"RI": 1.5165, "Na": 14.38, "Mg": 0.0, "Al": 1.94, "Si": 73.61, "K": 0.0, "Ca": 8.48, "Ba": 1.57, "Fe": 0.0},
        {"RI": 1.5210, "Na": 13.64, "Mg": 4.49, "Al": 1.10, "Si": 71.78, "K": 0.06, "Ca": 8.75, "Ba": 0.0, "Fe": 0.0}
    ])
    st.download_button(
        label="📥 Download Template CSV",
        data=sample_template.to_csv(index=False),
        file_name="glass_samples_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Uploaded Batch Preview:**")
            st.dataframe(batch_df.head(10))

            if st.button("Run Batch Prediction"):
                results_df = predict(model, batch_df)
                combined = pd.concat([batch_df, results_df], axis=1)
                
                st.success(f"Successfully processed {len(combined)} samples!")
                st.dataframe(combined)

                st.download_button(
                    label="💾 Download Prediction Results CSV",
                    data=combined.to_csv(index=False),
                    file_name="glass_forensics_predictions.csv",
                    mime="text/csv"
                )
        except Exception as ex:
            st.error(f"Error processing CSV file: {ex}")
