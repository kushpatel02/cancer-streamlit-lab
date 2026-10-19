"""Shared helpers for all dashboard pages."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
START_HINT = "Start the backend: `cd fastapi_app/src && python train.py && uvicorn main:app --reload`"


def backend_online() -> bool:
    try:
        return requests.get(f"{API_URL}/", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def get_model_info() -> dict:
    r = requests.get(f"{API_URL}/model/info", timeout=5)
    r.raise_for_status()
    return r.json()


def sidebar_status():
    """Render backend + model status in the sidebar. Returns model info, or None if unavailable."""
    with st.sidebar:
        if not backend_online():
            st.error("Backend offline 😱")
            st.caption(f"Expected at {API_URL}")
            return None
        try:
            info = get_model_info()
        except requests.exceptions.RequestException:
            st.warning("Backend online, but model not trained ⚠️")
            return None
        st.success("Backend online ✅")
        st.caption(f"Model: {info['model_type']}")
        m = info["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Acc", f"{m['accuracy']:.3f}")
        c2.metric("F1", f"{m['f1']:.3f}")
        c3.metric("AUC", f"{m['roc_auc']:.3f}")
        return info


def label(feature: str) -> str:
    return feature.replace("_", " ").capitalize()
