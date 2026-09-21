"""
MediScope API
-------------
FastAPI backend serving the disease-prediction / recommendation engine and
the web frontend.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /                 -> frontend (index.html)
    GET  /api/symptoms     -> list of all known symptoms (for the UI's picker)
    POST /api/predict      -> {"symptoms": [...]} -> full recommendation
    GET  /api/health       -> liveness + model metadata
"""

import sys
from pathlib import Path
from typing import List

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommend import RecommendationEngine  # noqa: E402

_engine: RecommendationEngine | None = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup instead of on the first request.
    get_engine()
    yield


app = FastAPI(
    title="MediScope API",
    description="Symptom-based disease prediction & personalized care recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=ROOT / "api" / "static"), name="static")


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., min_length=1, examples=[["itching", "skin_rash"]])
    top_k: int = Field(3, ge=1, le=41, description="Number of ranked predictions to return (1-41).")


@app.get("/")
def serve_frontend():
    return FileResponse(ROOT / "api" / "templates" / "index.html")


@app.get("/api/health")
def health():
    import json

    with open(ROOT / "models" / "metrics.json") as f:
        metrics = json.load(f)
    return {
        "status": "ok",
        "model": metrics["best_model"],
        "n_classes": metrics["n_classes"],
        "n_training_records": metrics["n_records"],
    }


@app.get("/api/symptoms")
def list_symptoms():
    engine = get_engine()
    return {"symptoms": engine.known_symptoms()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    engine = get_engine()
    try:
        result = engine.predict(req.symptoms, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
