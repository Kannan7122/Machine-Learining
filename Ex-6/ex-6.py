import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

data=pd.read_csv("ex-6.csv")
data.shape

feature = ['age', 'income', 'student' , 'credit_rating']
x = data[feature]
y = data.class_buy_computer

pca = PCA(n_components=2)
reg = LinearRegression()
pipeline = Pipeline(steps=[('pca', pca),('reg', reg)])
pipeline.fit(x, y)
pred = pipeline.predict(x)

print("Number of features before PCR:", x.shape[1])
print("Number of features after PCR:", pca.n_components_)
print("MAE",mean_absolute_error(y,pred))
print("MSE",mean_squared_error(y,pred))
print("RMSE",np.sqrt(mean_squared_error(y,pred)))
print("R^2",pipeline.score(x, y))
