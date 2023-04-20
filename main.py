from fastapi import FastAPI
from app.models.heuristic import Heuristic
from app.models.randomforest import RandomForest
from app.models.logisticregression import LogisticRegression
import numpy as np

from app.dto.predictrequest import PredictRequest
from tools.modeltypeenum import ModelTypeEnum

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
