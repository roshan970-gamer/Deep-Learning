import numpy as np, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

f=lambda x:np.cos(1.5*np.pi*x)
X=np.sort(np.random.rand(30)); y=f(X)+0.1*np.random.randn(30)

for i,d in enumerate([1,4,15]):
    plt.subplot(1,3,i+1)
    m=Pipeline([('p',PolynomialFeatures(degree=d,include_bias=False)),
                ('l',LinearRegression())])
    m.fit(X[:,None],y)
    t=np.linspace(0,1,100)
    plt.plot(t,m.predict(t[:,None])); plt.plot(t,f(t))
    plt.scatter(X,y,s=10); plt.title(f"d={d}")
    plt.xticks([]); plt.yticks([])
plt.show()
