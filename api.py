from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

model = joblib.load('model.pkl')

@app.get("/predict")
def predict(years:int):
    years = np.array(years)
    years = years.reshape(-1,1)
    prediction = model.predict(years)
    return {"Predicted Salary": float(prediction[0])}

