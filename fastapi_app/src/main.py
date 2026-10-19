from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from data import FEATURES
from predict import get_metadata, predict_many

app = FastAPI(title="Breast Cancer Diagnosis API", version="1.0")

# 30 features -> build the request schema from the feature list; all measurements must be >= 0
TumorFeatures = create_model("TumorFeatures", **{f: (float, Field(..., ge=0)) for f in FEATURES})


class BatchRequest(BaseModel):
    samples: list[TumorFeatures] = Field(min_length=1, max_length=500)


class Prediction(BaseModel):
    label: int
    diagnosis: str
    confidence: float
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    count: int
    predictions: list[Prediction]


def _run(fn, *args):
    try:
        return fn(*args)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not found. Run `python train.py` first.")


@app.get("/")
async def health():
    return {"status": "healthy"}


@app.get("/model/info")
async def model_info():
    return _run(get_metadata)


@app.post("/predict", response_model=Prediction)
async def predict(sample: TumorFeatures):
    return _run(predict_many, [sample.model_dump()])[0]


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    preds = _run(predict_many, [s.model_dump() for s in req.samples])
    return {"count": len(preds), "predictions": preds}
