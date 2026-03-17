from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI()

model = joblib.load('model.pkl')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(years:int):
    years = np.array(years)
    years = years.reshape(-1,1)
    prediction = model.predict(years)
    return {"prediction": float(prediction[0])}

