from fastapi import FastAPI, UploadFile, File
import numpy as np
import pandas as pd
from io import BytesIO

from app.dto.predict_request import PredictRequest
from app.dto.model_type_enum import ModelTypeEnum

description = """
Application with an API exposing four models making predictions of <a href="https://archive.ics.uci.edu/ml/datasets/covertype">cover type in the forest</a>.
Models:
1. Heuristic - assigns the type of cover most frequently occurring at a given elevation
2. Logistic regression - features selected using mutual information
3. Random forest - 50 estimators, maximum depth 30, features selected using mutual information
4. Neural network - 3 hidden dense layers, hyperparameters tuned with Optuna (a Bayesian optimization algorithm)
"""

tags_metadata = [
    {
        "name": "models",
        "description": "Access to models",
    },
]

app = FastAPI(
    title="Forest cover type prediction models",
    version="1.0.0",
    description=description,
    contact={
        "name": "Jan Sowa",
        "url": "http://www.github.com/jansowa/",
        "email": "jan.piotr.sowa@gmail.com",
    },
    openapi_tags=tags_metadata
)


@app.post("/model", description="Method for single sample prediction using selected model", tags=["models"],
          summary="Predict single sample")
async def predict_single_sample(request_body: PredictRequest):
    model_type: ModelTypeEnum = request_body.type
    model = model_type.get_model_by_type()
    return {
        "prediction": np.array(model.predict(np.array(request_body.values).reshape(1, -1))).tolist()
    }


@app.post("/model/file", description="Load csv.file and return predictions for all samples using selected model", tags=["models"],
          summary="Predict from csv")
async def predict_from_file(model_type: ModelTypeEnum = ModelTypeEnum.random_forest,
                            csv_file: UploadFile = File(description="One sample (54 features) in each line")):
    contents = csv_file.file.read()
    data = BytesIO(contents)
    df = pd.read_csv(data, header=None)
    data.close()
    csv_file.file.close()
    model = model_type.get_model_by_type()

    return {
        "predictions": np.array(model.predict(df.values[:, :54])).tolist()
    }


predict_from_file.__doc__ = """
    Parameters:
        - `model_type`: Selected model type. Possible values: \"heuristic\", "
                                            "\"logistic_regression\", \"random_forest\", \"neural_network\""
    """