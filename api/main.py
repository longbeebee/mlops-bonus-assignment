import os
from typing import Annotated

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sklearn-iris-classifier")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
app = FastAPI(title="Iris MLOps API", version="1.0.0")
model = None


class IrisRequest(BaseModel):
    features: Annotated[list[float], Field(min_length=4, max_length=4)]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS, "loaded": model is not None}


def get_model():
    global model
    if model is None:
        try:
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"model alias is not ready: {exc}") from exc
    return model


@app.post("/predict")
def predict(payload: IrisRequest):
    loaded = get_model()
    feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    prediction = loaded.predict(pd.DataFrame([payload.features], columns=feature_names))
    return {"prediction": int(prediction[0]), "model": MODEL_NAME, "alias": MODEL_ALIAS}
