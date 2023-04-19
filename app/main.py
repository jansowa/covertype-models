from fastapi import FastAPI
from models.heuristic import Heuristic

app = FastAPI()

@app.get("/model")
async def root(values: str):
    heuristic = Heuristic()
    values = [float(val) for val in values.split(",")]
    prediction = heuristic.predict(values)
    return {"prediction": prediction}
