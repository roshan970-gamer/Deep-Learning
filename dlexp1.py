import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

a = ['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog']
p = ['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog']

cm = confusion_matrix(a, p)
sns.heatmap(cm, annot=True, cmap='RdPu',
            xticklabels=['Dog','Not Dog'],
            yticklabels=['Dog','Not Dog'])

plt.show()

