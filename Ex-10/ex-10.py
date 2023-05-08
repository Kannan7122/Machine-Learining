import numpy as nm
import pandas as pd

data = pd.read_csv("ex-10.csv")
features =['age','income','student','credit']
x= data[features]
y = data.buy

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=23)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators = 10, learning_rate=0.05,max_features=3,random_state=100)
gb.fit(xtr,ytr)
pred = gb.predict(xte)

from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(y,gb.predict(x))
print("Confusion Matrix")
print(cm)
print("Classification Report")
print(classification_report(yte,pred))
