import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

X,y = load_digits(return_X_y=True)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=23)

p = RandomForestClassifier().fit(Xtr,ytr).predict(Xte)

print("Acc:", accuracy_score(yte,p))
sns.heatmap(confusion_matrix(yte,p), annot=True, fmt='g')
plt.show()
