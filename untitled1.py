# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:49:02 2019

@author: Mohammad_Younesi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns

scaler = MinMaxScaler()
scaler1 = StandardScaler()
scaler2 = QuantileTransformer()
#scaler3 = PowerTransformer()
train_x = np.loadtxt("train_x")
train_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")
test_y = np.loadtxt("test_y")

#in case of reducing second error type
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)
#train_x = scaler1.fit_transform(train_x)
#test_x = scaler1.fit_transform(test_x)
#train_x = scaler2.fit_transform(train_x)
#test_x = scaler2.fit_transform(test_x)
#train_x = scaler3.fit_transform(train_x)
#test_x = scaler3.fit_transform(test_x)

model1 = xgboost.XGBClassifier()
model21 = svm.SVC(class_weight={0:0.8, 1:0.2}) #optional: probability=True
model22 = svm.SVC(kernel = "rbf")
model23 = svm.SVC(kernel = "linear")
model24 = svm.SVC(kernel = "poly")
model25 = svm.SVC(kernel = "sigmoid")
model3 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=8) #criterion="entropy", max_depth
model4 = RandomForestClassifier()
model5 = LinearDiscriminantAnalysis()
model6 =  QuadraticDiscriminantAnalysis()
model7 = KNeighborsClassifier(n_neighbors=9) #n_neighbors=k
#model8 = MultinomialNB()
#U can change the solver to ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
#also U can change class weight to change the error types
# U can change the threshold by adding class_weight={0:0.8, 1:0.2}
logreg = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000)

model = logreg #U cang change the model to all the models above


model.fit(train_x,train_y)
y_pred = model.predict(test_x)
conf_mat = metrics.confusion_matrix(test_y,y_pred)
error1 = conf_mat[0][1]/(conf_mat[0][0]+conf_mat[0][1])
error2 = 1-metrics.recall_score(test_y, y_pred)


print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
print("Precision:",metrics.precision_score(test_y, y_pred))
print("Recall:",metrics.recall_score(test_y, y_pred))
print("Error type 1:",error1)
print("Error type 2:",error2)


y_pred_proba = model.predict_proba(test_x)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_y,  y_pred_proba)
auc = metrics.roc_auc_score(test_y, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#this section is for Random Forest Classifier
#Here we can find important features
#feature_imp = pd.Series(model.feature_importances_).sort_values(ascending=False)
#new_train_x = train_x[:,feature_imp.index[0:50]] #selecting 50 most important features
#new_train_y = train_y
#new_test_x = test_x[:,feature_imp.index[0:50]] #selecting 50 most important features
#new_test_y = test_y
#model.fit(new_train_x,new_train_y)
#new_y_pred = model.predict(new_test_x)
#new_conf_mat = metrics.confusion_matrix(new_test_y,new_y_pred)
#error1 = new_conf_mat[0][1]/(new_conf_mat[0][0]+new_conf_mat[0][1])
#error2 = 1-metrics.recall_score(new_test_y, new_y_pred)
#
#
#print("Accuracy:",metrics.accuracy_score(new_test_y, new_y_pred))
#print("Precision:",metrics.precision_score(new_test_y, new_y_pred))
#print("Recall:",metrics.recall_score(new_test_y, new_y_pred))
#print("Error type 1:",error1)
#print("Error type 2:",error2)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(conf_mat), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(test_y, y_pred_proba) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])


