"""FastAPI endpoint for the preference model (same core as the Streamlit demo).

  uvicorn demo.api:app --host 127.0.0.1 --port 8000
  curl -s localhost:8000/predict -H 'Content-Type: application/json' \
       -d '{"prompt": "How do I boil an egg?", "response_a": "...", "response_b": "..."}'
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import DEFAULT_MODEL, MODELS, predict  # noqa: E402

app = FastAPI(title="Pairwise Preference Model",
              description="P(response A preferred) with calibrated confidence and bias checks.")


class PairRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    response_a: str = Field(..., min_length=1)
    response_b: str = Field(..., min_length=1)
    model: str = Field(MODELS[DEFAULT_MODEL], description="run name, see GET /models")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"default": MODELS[DEFAULT_MODEL], "models": {run: label for label, run in MODELS.items()}}


@app.post("/predict")
def predict_pair(req: PairRequest):
    if req.model not in MODELS.values():
        raise HTTPException(400, f"unknown model {req.model!r}; see GET /models")
    return predict(req.prompt, req.response_a, req.response_b, req.model)
