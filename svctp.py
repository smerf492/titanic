import numpy as np
import pandas as pd
from random import randint
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv', sep=",")
data.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1,inplace=True)
data.replace(["male","female"],[0,1],inplace=True)
data['Fare']=data['Fare'].fillna(0)

datak= pd.read_csv('test.csv', sep=",")
datak.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1,inplace=True)
datak.replace(["male","female"],[0,1],inplace=True)
datak['Fare']=datak['Fare'].fillna(0)  #jeden elemant

X=data.drop(["Survived"],axis=1)
y=data["Survived"]

#Ran=randint(1,1000)
Ran=421

def meaner(df):
 ma=df[df['Sex']==0]['Age'].mean()
 mb=df[df['Sex']==1]['Age'].mean()
 df[df['Sex']==0]['Age']=df[df['Sex']==0]['Age'].fillna(ma)
 df['Age']=df['Age'].fillna(mb)
 return df

pd.options.mode.chained_assignment = None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Ran)

X_train=meaner(X_train)
X_test=meaner(X_test)
datak=meaner(datak)

kfold = StratifiedKFold(n_splits=5)

pipe = Pipeline([('pre', StandardScaler()), ('clas', svm.SVC())])

param_grid = {'pre': [StandardScaler(), None],
              'clas__C': [150, 200, 260],
              'clas__gamma': [0.015, 0.02, 0.025]}

grid_1 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)
grid_1.fit(X_train, y_train)
print(grid_1.best_params_)
print("accuracy_train: {}".format(metrics.accuracy_score(y_train, grid_1.predict(X_train))))
print("accuracy_test:  {}".format(metrics.accuracy_score(y_test, grid_1.predict(X_test))))
print(Ran)
t=grid_1.predict(datak)
w=[]
lp=892
for l in t:
 w.append([lp,l])
 lp=lp+1
dataw = pd.DataFrame(w, columns=["PassengerId","Survived"])
dataw.to_csv('wyn.csv',sep=',',index=False)

