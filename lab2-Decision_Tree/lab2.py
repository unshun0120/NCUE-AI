import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, tree
#from sklearn.cross_validation import train_test_split #舊版
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

order = pd.read_csv("train.csv")

#Label encoding : 把某個class底下的label map成某個數字
#將以下欄位轉成數字 1st, 2nd, ...
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(order["alt"])
encoded_class1 = label_encoder.fit_transform(order["bar"]) 
encoded_class2 = label_encoder.fit_transform(order["fri"])
encoded_class3 = label_encoder.fit_transform(order["hun"])   
encoded_class4 = label_encoder.fit_transform(order["pat"]) 
encoded_class5 = label_encoder.fit_transform(order["price"])
encoded_class6 = label_encoder.fit_transform(order["rain"])
encoded_class7 = label_encoder.fit_transform(order["res"])   
encoded_class8 = label_encoder.fit_transform(order["est"])  

#將type欄的french thai burger italian 轉成 0, 1, 2, 3
order["type"].replace(['french', 'thai', 'burger', 'italian'], [0, 1, 2, 3], inplace=True) 
X = pd.DataFrame([order["type"], encoded_class, encoded_class1, encoded_class2, encoded_class3, encoded_class4, encoded_class5, encoded_class6, encoded_class7, encoded_class8]).T
X.columns = ["type", "alt", "bar", "fri", "hun", "pat", "price", "rain", "res", "est"]
y = order["willWait"]

#開始做Decision Tree
Xtrain, XTest, yTrain, yTest = \
train_test_split(X, y, test_size=0.25, random_state=1)
dtree = tree.DecisionTreeClassifier()
dtree.fit(Xtrain, yTrain)
print("準確率 :", dtree.score(XTest, yTest))
preds = dtree.predict_proba(X=XTest)
print(pd.crosstab(preds[:,0], columns=[XTest["type"], X["alt"], X["bar"], X["fri"], X["hun"], X["pat"], X["price"], X["rain"], X["res"], X["est"]]))

#Draw Decision Tree
plt.figure(figsize=(8,6))   #figure size = 8*6
plot_tree(dtree, filled=True, fontsize=8)   #字型大小調成8
plt.show()