import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

samples_per_class = 5
figure = plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))

idx_cls = 0
for cls in classes:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs, samples_per_class, replace=False)
  i = 0
  for idx in idxs:
    plt_idx = i * nclasses + idx_cls + 1
    p = plt.subplot(samples_per_class, nclasses, plt_idx);
    p = sns.heatmap(np.array(X.loc[idx]).reshape(28,28), cmap=plt.cm.gray,
             xticklabels=False, yticklabels=False, cbar=False);
    p = plt.axis('off');
    i += 1
  idx_cls += 1

idxs = np.flatnonzero(y == '0')
print(np.array(X.loc[idxs[0]]))

print(len(X))
print(len(X.loc[0]))

 X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 9, train_size = 7500, test_size = 2500)
 X_train_scale = X_train/255
 X_test_scale = X_test/255

print(X.loc[0])
print(y.loc[0])

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scale, y_train)

y_pred = clf.predict(X_test_scale)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cm = pd.crosstab(y_test, y_pred, rownames = ["actual"], colnames = ["predicted"])
p = plt.figure(figsize = (10, 10))
p = sns.heatmap(cm,annot = True, fmt = "d", cbar = False)

