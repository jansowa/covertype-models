from fastapi import FastAPI
from models.heuristic import Heuristic
from pydantic import BaseModel
from enum import Enum

class ModelTypeEnum(str, Enum):
    heuristic = "heuristic"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    neural_network = "neural_network"

class PredictRequest(BaseModel):
    values: list[float]
    type: ModelTypeEnum

app = FastAPI()

@app.post("/model")
async def predict(request_body: PredictRequest):
    if request_body.type == ModelTypeEnum.heuristic:
        model = Heuristic()
    else:
        return {
            "error": "Not implemented yet"
        }
    return {
        "prediction": model.predict(request_body.values)
    }