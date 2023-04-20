from fastapi import FastAPI
from app.models.heuristic import Heuristic
from app.models.randomforest import RandomForest
from app.models.logisticregression import LogisticRegression
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class ModelTypeEnum(str, Enum):
    heuristic = "heuristic"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    neural_network = "neural_network"


class PredictRequest(BaseModel):
    values: list[float] = Field(title="Values for single prediction", min_items=54, max_items=54)
    type: ModelTypeEnum = Field(title="Model type",
                                description="Possible values - \"heuristic\", "
                                            "\"logistic_regression\", \"random_forest\", \"neural_network\"")


app = FastAPI()


@app.post("/model", description="Method for single sample prediction")
async def predict(request_body: PredictRequest):
    if request_body.type == ModelTypeEnum.heuristic:
        model = Heuristic()
    elif request_body.type == ModelTypeEnum.random_forest:
        model = RandomForest()
    elif request_body.type == ModelTypeEnum.logistic_regression:
        model = LogisticRegression()
    else:
        return {
            "error": "Not implemented yet"
        }
    return {
        "prediction": np.array(model.predict(np.array(request_body.values).reshape(1, -1))).tolist()
    }
