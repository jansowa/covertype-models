from pydantic import BaseModel, Field

from app.dto.model_type_enum import ModelTypeEnum


class PredictRequest(BaseModel):
    values: list[float] = Field(title="Values for single prediction", min_items=54, max_items=54)
    type: ModelTypeEnum = Field(title="Model type",
                                description="Possible values - \"heuristic\", "
                                            "\"logistic_regression\", \"random_forest\", \"neural_network\"")
