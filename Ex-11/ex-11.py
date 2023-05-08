import numpy as num
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv('ex-11.csv')
colnames=['RID','Age','Income','Student','Credit_Rating','Buy_Computer']
features=['Income','Age','Credit_Rating','Student']
x=data[features]
y=data.Buy_Computer
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=3,max_depth=2,max_features='auto',bootstrap=True)
rf.fit(xtr,ytr)
pred=rf.predict(xte)

from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y,rf.predict(x))
print(classification_report(y,rf.predict(x)))
print(cm)
print("Classification Report")
print(classification_report(y,rf.predict(x)))

from sklearn import tree
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(7,7),facecolor='w')
tree.plot_tree(rf.estimators_[0],rounded=True,feature_names=features,class_names=y,filled=True,fontsize=12)
plt.show()
tree.plot_tree(rf.estimators_[1],rounded=True,feature_names=features,class_names=y,filled=True,fontsize=12)
plt.show()
tree.plot_tree(rf.estimators_[2],rounded=True,feature_names=features,class_names=y,filled=True,fontsize=12)
plt.show()
