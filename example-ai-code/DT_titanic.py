import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, tree
#from sklearn.cross_validation import train_test_split #舊版
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

titanic = pd.read_csv("train.csv")
#Age中有NaN資料，計算age中位數
age_median = np.nanmedian(titanic["Age"]) 
#若空以中位數取代
new_age = np.where(titanic["Age"].isnull(), age_median, titanic["Age"])
titanic["Age"] = new_age

#PClass欄位為無\文字轉數字
#Label encoding : 把某個class底下的label map成某個數字
#將艙等轉成數字 1st, 2nd, ...
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(titanic["Pclass"]) 

#將female male 轉成 0, 1
titanic["Sex"].replace(['female','male'],[0,1],inplace=True) 
X = pd.DataFrame([titanic["Sex"], encoded_class]).T
#Sex為string 
X.columns = ["Sex", "Pclass"]
#X= pd.DataFrame([encoded_class,  titanic["Age"]]).T
y = titanic["Survived"]

#開始做Decision Tree
Xtrain, XTest, yTrain, yTest = \
train_test_split(X, y, test_size=0.25, random_state=1)
dtree = tree.DecisionTreeClassifier()
dtree.fit(Xtrain, yTrain)
print("準確率 :", dtree.score(XTest, yTest))
preds = dtree.predict_proba(X=XTest)
print(pd.crosstab(preds[:,0], columns=[X["Pclass"],XTest["Sex"]]))

#Draw Decision Tree
plt.figure()
plot_tree(dtree, filled=True)
plt.show()