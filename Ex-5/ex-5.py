import numpy as nm
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("ex-5.csv")
colnames =['sno','age','income','student','credit','buy']
features =['age','income','credit']
x= data[features]
y = data.buy
xtr, xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state =0)

clf = DecisionTreeClassifier(criterion = "entropy",max_depth=3)
clf = clf.fit(xtr,ytr)
ypre = clf.predict(xte)

from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(y,clf.predict(x))
print("Confusion Matrix")
print(cm)
print("Classification Report")
print(classification_report(y,clf.predict(x)))

from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6),facecolor='w')
a = tree.plot_tree(clf,rounded = True,feature_names = features,class_names =y,filled = True,fontsize = 9)
plt.show()
