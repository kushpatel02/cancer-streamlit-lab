import pandas as pd
import streamlit as st

from common import START_HINT, label, sidebar_status

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")
st.write("# Model Insights 📊")

info = sidebar_status()
if info is None:
    st.info(START_HINT)
    st.stop()

m = info["metrics"]
c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{m['accuracy']:.2%}")
c2.metric("F1 score", f"{m['f1']:.3f}")
c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

st.subheader("Top 10 most important features")
imp = pd.Series(info["feature_importances"]).sort_values(ascending=False).head(10)
st.bar_chart(imp.rename(index=label), horizontal=True)

st.subheader("Compare classes")
feature = st.selectbox("Feature", info["features"], format_func=label)
medians = pd.Series({c.capitalize(): info["class_medians"][c][feature] for c in info["classes"]},
                    name="median")
st.bar_chart(medians)
lo, hi, med = info["feature_ranges"][feature]
st.caption(f"Dataset range: {lo} – {hi} (overall median {med})")
