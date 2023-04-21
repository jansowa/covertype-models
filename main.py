from fastapi import FastAPI
from app.models.heuristic import Heuristic
from app.models.randomforest import RandomForest
from app.models.logisticregression import LogisticRegression
from app.models.neuralnetwork import NeuralNetwork
from app.models.abstractmodel import AbstractModel
import numpy as np

from app.dto.predictrequest import PredictRequest
from app.dto.modeltypeenum import ModelTypeEnum

app = FastAPI()


@app.post("/model", description="Method for single sample prediction")
async def predict(request_body: PredictRequest):
    model: AbstractModel
    if request_body.type == ModelTypeEnum.heuristic:
        model = Heuristic()
    elif request_body.type == ModelTypeEnum.random_forest:
        model = RandomForest()
    elif request_body.type == ModelTypeEnum.logistic_regression:
        model = LogisticRegression()
    elif request_body.type == ModelTypeEnum.neural_network:
        model = NeuralNetwork()
    else:
        return {
            "error": "Not implemented yet"
        }
    return {
        "prediction": np.array(model.predict(np.array(request_body.values).reshape(1, -1))).tolist()
    }
