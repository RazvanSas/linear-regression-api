import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

regressor = LinearRegression()

regressor.fit(X, y)

joblib.dump(regressor, 'model.pkl')