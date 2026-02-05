import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

X,y = load_wine(return_X_y=True)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=1)

p = KNeighborsClassifier(5).fit(Xtr,ytr).predict(Xte)

print("Acc:", accuracy_score(yte,p))
sns.heatmap(confusion_matrix(yte,p), annot=True)
plt.show()
