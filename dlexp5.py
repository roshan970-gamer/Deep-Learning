import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X,y = load_iris(return_X_y=True)
X,y = X[:,0:1], X[:,1]

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)
p = LinearRegression().fit(Xtr,ytr).predict(Xte)

print("MSE:", mean_squared_error(yte,p))
plt.scatter(Xte,yte); plt.plot(Xte,p); plt.show()
print("Pred:", LinearRegression().fit(X,y).predict([[5]])[0])
