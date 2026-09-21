"""
tests/test_api.py
------------------
Basic automated tests for the MediScope API. Run with:

    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_classes"] == 41


def test_symptoms_list_nonempty():
    r = client.get("/api/symptoms")
    assert r.status_code == 200
    symptoms = r.json()["symptoms"]
    assert len(symptoms) == 132
    assert "itching" in symptoms


def test_predict_known_symptoms():
    r = client.post("/api/predict", json={"symptoms": ["itching", "skin_rash"]})
    assert r.status_code == 200
    body = r.json()
    assert body["disease"]
    assert body["description"]
    assert isinstance(body["medications"], list)
    assert isinstance(body["precautions"], list)


def test_predict_every_disease_has_full_recommendation():
    """Every disease class must resolve against description/medication/diet/
    precaution/workout tables — regression test for the data-cleaning fixes
    in src/recommend.py (whitespace + 'diseae'/'disease' typo)."""
    import pandas as pd

    from src.recommend import RecommendationEngine

    engine = RecommendationEngine()
    train = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "Training.csv")

    for disease in train["prognosis"].unique():
        assert engine._lookup_disease_field(engine.description, "Disease", disease) is not None, disease
        assert engine._lookup_disease_field(engine.medications, "Disease", disease) is not None, disease
        assert engine._lookup_disease_field(engine.diets, "Disease", disease) is not None, disease
        assert engine._lookup_disease_field(engine.precautions, "Disease", disease) is not None, disease


def test_predict_rejects_empty_symptom_list():
    r = client.post("/api/predict", json={"symptoms": []})
    assert r.status_code == 422


def test_predict_rejects_all_unknown_symptoms():
    r = client.post("/api/predict", json={"symptoms": ["not_a_real_symptom"]})
    assert r.status_code == 400


def test_predict_tolerates_mixed_known_unknown_symptoms():
    r = client.post(
        "/api/predict", json={"symptoms": ["itching", "not_a_real_symptom"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["resolved_symptoms"] == ["itching"]
    assert body["unresolved_symptoms"] == ["not_a_real_symptom"]


def test_predict_rejects_invalid_top_k():
    """Regression test: top_k=0 used to crash with an unhandled 500
    (IndexError) instead of a clean validation error."""
    r = client.post("/api/predict", json={"symptoms": ["itching"], "top_k": 0})
    assert r.status_code == 422

    r = client.post("/api/predict", json={"symptoms": ["itching"], "top_k": -5})
    assert r.status_code == 422


def test_predict_caps_oversized_top_k():
    r = client.post("/api/predict", json={"symptoms": ["itching"], "top_k": 9999})
    assert r.status_code == 422  # now rejected outright (max is 41 classes)


def test_predict_deduplicates_repeated_symptoms():
    """Regression test: submitting the same symptom multiple times (or in
    different casing) used to show up duplicated in resolved_symptoms."""
    r = client.post(
        "/api/predict", json={"symptoms": ["itching", "Itching", "itching"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["resolved_symptoms"] == ["itching"]


def test_frontend_serves():
    r = client.get("/")
    assert r.status_code == 200
    assert b"MediScope" in r.content
