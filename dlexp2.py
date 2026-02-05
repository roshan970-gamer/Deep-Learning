import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

w = load_wine()
d = pd.DataFrame(w.data, columns=w.feature_names)
d['T'] = w.target
d = d[d['T'] != 2]          # FIX HERE

X, y = d.drop('T', axis=1), d['T']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=1)

p = DecisionTreeClassifier().fit(Xtr,ytr).predict(Xte)

print("Acc:", accuracy_score(yte,p))
sns.heatmap(confusion_matrix(yte,p), annot=True)
plt.show()
 
