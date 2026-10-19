import json

import pandas as pd
import requests
import streamlit as st

from common import API_URL, START_HINT, label, sidebar_status

st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon="🩺", layout="wide")
st.write("# Breast Cancer Diagnosis 🩺")
st.caption("Predict whether a tumor is malignant or benign from 30 cell-nucleus measurements.")

info = sidebar_status()
if info is None:
    st.info(START_HINT)
    st.stop()

features, ranges = info["features"], info["feature_ranges"]
GROUPS = {
    "Mean values": lambda f: f.startswith("mean_"),
    "Standard error": lambda f: f.endswith("_error"),
    "Worst values": lambda f: f.startswith("worst_"),
}

# Slider values live in session_state (key = feature name) so presets can overwrite them
for f in features:
    st.session_state.setdefault(f, float(ranges[f][2]))
st.session_state.setdefault("history", [])


def load_preset(name: str):
    values = {f: ranges[f][2] for f in features} if name == "median" else info["class_medians"][name]
    for f in features:
        st.session_state[f] = float(values[f])


# ---------------- Input ----------------
mode = st.sidebar.radio("Input mode", ["Sliders", "Upload JSON"])
payload = None

if mode == "Sliders":
    st.write("**Quick fill**")
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    c1.button("Typical malignant", on_click=load_preset, args=("malignant",))
    c2.button("Typical benign", on_click=load_preset, args=("benign",))
    c3.button("Reset to median", on_click=load_preset, args=("median",))

    for tab, (name, match) in zip(st.tabs(list(GROUPS)), GROUPS.items()):
        with tab:
            cols = st.columns(2)
            for i, f in enumerate(f for f in features if match(f)):
                lo, hi, _ = ranges[f]
                cols[i % 2].slider(label(f), float(lo), float(hi), step=(hi - lo) / 100, key=f, format="%.4f")
    payload = {f: st.session_state[f] for f in features}

else:
    file = st.file_uploader("Upload a JSON file (see data/test.json)", type=["json"])
    if file:
        try:
            data = json.load(file)
            data = data.get("input_test", data)  # accept wrapped or flat format
            missing = [f for f in features if f not in data]
            if missing:
                st.error(f"Missing {len(missing)} feature(s): {', '.join(missing[:5])}...")
            else:
                payload = {f: data[f] for f in features}
                with st.expander("Preview file"):
                    st.json(payload)
        except json.JSONDecodeError:
            st.error("Could not parse the file as JSON.")

# ---------------- Prediction ----------------
if st.button("Predict", type="primary", disabled=payload is None):
    try:
        with st.spinner("Predicting..."):
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.RequestException:
        st.toast(":red[Backend unreachable. Refresh and check backend status]", icon="🔴")
    else:
        if r.status_code == 200:
            res = r.json()
            diagnosis = res["diagnosis"].capitalize()
            msg = f"Prediction: **{diagnosis}** ({res['confidence']:.1%} confidence)"
            (st.error if res["diagnosis"] == "malignant" else st.success)(msg)
            st.bar_chart(pd.Series(res["probabilities"], name="probability"), horizontal=True)
            st.session_state.history.append(
                {"source": mode, "diagnosis": diagnosis, "confidence": res["confidence"]})
        elif r.status_code == 422:
            st.error("Input rejected by API validation (values must be numbers ≥ 0).")
        else:
            st.toast(f":red[Server returned {r.status_code}]", icon="🔴")

# ---------------- History ----------------
if st.session_state.history:
    st.subheader("Prediction history (this session)")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
