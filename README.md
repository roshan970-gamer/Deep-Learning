# Deep-Learning
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix

a = ['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog']
p = ['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog']

sns.heatmap(confusion_matrix(a,p), annot=True, cmap='RdPu',
            xticklabels=['Dog','Not Dog'], yticklabels=['Dog','Not Dog'])
plt.show()
