import pandas as pd
import requests
import streamlit as st

from common import API_URL, START_HINT, sidebar_status

st.set_page_config(page_title="Batch Prediction", page_icon="📦", layout="wide")
st.write("# Batch Prediction 📦")
st.caption("Upload a CSV with one row per sample (max 500 rows).")

info = sidebar_status()
if info is None:
    st.info(START_HINT)
    st.stop()

features = info["features"]
st.download_button("Download CSV template", ",".join(features) + "\n", "template.csv", "text/csv")

file = st.file_uploader("Upload CSV (see data/batch_test.csv)", type=["csv"])
if file:
    df = pd.read_csv(file)
    missing = sorted(set(features) - set(df.columns))
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    if len(df) > 500:
        st.error(f"{len(df)} rows uploaded; the API accepts at most 500.")
        st.stop()
    st.write(f"Loaded **{len(df)}** rows")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Run batch prediction", type="primary"):
        try:
            with st.spinner(f"Scoring {len(df)} rows..."):
                r = requests.post(f"{API_URL}/predict/batch",
                                  json={"samples": df[features].to_dict(orient="records")}, timeout=30)
        except requests.exceptions.RequestException:
            st.error("Backend unreachable.")
            st.stop()
        if r.status_code != 200:
            st.error(f"Server returned {r.status_code}. Check for empty or negative values.")
            st.stop()

        preds = pd.DataFrame(r.json()["predictions"])
        out = df.assign(diagnosis=preds["diagnosis"], confidence=preds["confidence"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", len(out))
        c2.metric("Malignant", int((out["diagnosis"] == "malignant").sum()))
        c3.metric("Benign", int((out["diagnosis"] == "benign").sum()))

        st.dataframe(out[["diagnosis", "confidence"] + features], use_container_width=True)
        st.download_button("Download results CSV", out.to_csv(index=False), "predictions.csv", "text/csv")
