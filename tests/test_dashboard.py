"""UI tests with Streamlit's AppTest. The backend is mocked, so no server is needed."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
META = json.loads((ROOT / "fastapi_app" / "model" / "metadata.json").read_text())


def fake_get(url, *args, **kwargs):
    resp = MagicMock(status_code=200)
    resp.json.return_value = META if url.endswith("/model/info") else {"status": "healthy"}
    return resp


def fake_post(url, *args, **kwargs):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"label": 1, "diagnosis": "benign", "confidence": 0.97,
                              "probabilities": {"malignant": 0.03, "benign": 0.97}}
    return resp


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_data.clear()


def run(at):
    with patch("requests.get", side_effect=fake_get), patch("requests.post", side_effect=fake_post):
        return at.run()


def test_backend_offline_shows_error():
    at = AppTest.from_file(str(SRC / "Dashboard.py"), default_timeout=30)
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
        at.run()
    assert not at.exception
    assert any("offline" in e.value for e in at.error)


def test_dashboard_renders_30_sliders():
    at = run(AppTest.from_file(str(SRC / "Dashboard.py"), default_timeout=30))
    assert not at.exception
    assert len(at.slider) == 30


def test_preset_fills_sliders():
    at = run(AppTest.from_file(str(SRC / "Dashboard.py"), default_timeout=30))
    next(b for b in at.button if b.label == "Typical malignant").click()
    run(at)
    assert at.slider(key="mean_radius").value == pytest.approx(META["class_medians"]["malignant"]["mean_radius"])


def test_predict_shows_result_and_history():
    at = run(AppTest.from_file(str(SRC / "Dashboard.py"), default_timeout=30))
    next(b for b in at.button if b.label == "Predict").click()
    run(at)
    assert any("Benign" in s.value for s in at.success)
    assert len(at.session_state["history"]) == 1


@pytest.mark.parametrize("page", ["1_Batch_Prediction.py", "2_Model_Insights.py"])
def test_pages_render(page):
    at = run(AppTest.from_file(str(SRC / "pages" / page), default_timeout=30))
    assert not at.exception
