from fastapi import FastAPI, UploadFile
import numpy as np
import pandas as pd
from io import BytesIO

from app.dto.predict_request import PredictRequest
from app.dto.model_type_enum import ModelTypeEnum

app = FastAPI()


@app.post("/model", description="Method for single sample prediction")
async def predict_single_sample(request_body: PredictRequest):
    model_type: ModelTypeEnum = request_body.type
    model = model_type.get_model_by_type()
    return {
        "prediction": np.array(model.predict(np.array(request_body.values).reshape(1, -1))).tolist()
    }


@app.post("/model/file", description="Load csv.file and return predictions for all samples")
async def predict_from_file(csv_file: UploadFile, model_type: ModelTypeEnum):
    contents = csv_file.file.read()
    data = BytesIO(contents)
    df = pd.read_csv(data, header=None)
    data.close()
    csv_file.file.close()
    model = model_type.get_model_by_type()

    return {
        "predictions": np.array(model.predict(df.values[:, :54])).tolist()
    }
