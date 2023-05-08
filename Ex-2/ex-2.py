import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
data= pd.read_csv('ex-2.csv')  
x= data.iloc[:, :-1].values  
y= data.iloc[:, 1].values

from sklearn.model_selection import train_test_split  
xtr, xte, ytr, yte= train_test_split(x, y, test_size= 1/3, random_state=0)

from sklearn import linear_model
regressor= linear_model.LinearRegression()  
regressor.fit(xtr, ytr)
yp= regressor.predict(xte)  
xp= regressor.predict(xtr)

from sklearn import metrics
print("MAE", metrics.mean_absolute_error(yte,yp))
print("MSE", metrics.mean_squared_error(yte,yp))
print("RMSE", nm.sqrt(metrics.mean_squared_error(yte,yp)))

mtp.scatter(xtr, ytr, color="green")   
mtp.plot(xtr, xp, color="red")    
mtp.title("Overs vs Runs (Training Dataset)")  
mtp.xlabel("Overs")  
mtp.ylabel("Runs")  
mtp.show()
