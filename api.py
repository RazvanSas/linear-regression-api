from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from pydantic import BaseModel
app = FastAPI()

model = joblib.load('model.pkl')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_methods=["*"],
    allow_headers=["*"],
)

class Data(BaseModel):
    years: int

@app.post("/predict")
def predict(data: Data):
    years = np.array(data.years)
    years = years.reshape(-1,1)
    prediction = model.predict(years)
    return {"prediction": float(prediction[0])}

