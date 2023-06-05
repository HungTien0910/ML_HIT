import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('ExML\loan_train.csv')

df.shape
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])

df['loan_status'].value_counts()
import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()

df.groupby(['education'])['loan_status'].value_counts(normalize=True)

df[['Principal','terms','age','Gender','education']].head()

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

X = Feature
X[0:5]

y = df['loan_status'].values
y[0:5]

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# Using KNN (K Nearest Neighbor)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
X_train, y_train = np.array(X_train), np.array(y_train)

k_values = range(1, 10)
accuracy = [accuracy_score(y_val, neighbors.KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_val)) for k in k_values]

best_k = k_values[np.argmax(accuracy)]
print("Best k:", best_k)
plt.plot(k_values, accuracy)
plt.show()

Kmean = neighbors.KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
y_pred_train = Kmean.predict(X_train)
y_pred = Kmean.predict(X_val)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Training accuracy:", train_accuracy)
train_accuracy_percentage = 100 * accuracy_score(y_train, y_pred_train)
val_accuracy_percentage = 100 * accuracy_score(y_val, y_pred)

print(f"Accuracy of KNN for training: {train_accuracy_percentage:.2f}%")
print(f"Accuracy of KNN for validation: {val_accuracy_percentage:.2f}%")

# Using Decision Tree

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
train_accuracy_percentage = 100 * accuracy_score(y_train, decision_tree.predict(X_train))
val_accuracy_percentage = 100 * accuracy_score(y_val, decision_tree.predict(X_val))

print(f"Accuracy of decision tree for training: {train_accuracy_percentage:.2f}%")
print(f"Accuracy of decision tree for validation: {val_accuracy_percentage:.2f}%")

# Using Support Vector Machine

from sklearn.svm import SVC

svm_model = SVC(C=1e5, kernel="linear").fit(X_train, y_train)
train_accuracy_percentage = 100 * accuracy_score(y_train, svm_model.predict(X_train))
val_accuracy_percentage = 100 * accuracy_score(y_val, svm_model.predict(X_val))

print(f"Accuracy of SVM model for training: {train_accuracy_percentage:.2f}%")
print(f"Accuracy of SVM model for validation: {val_accuracy_percentage:.2f}%")

# Using Logistic Regression

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(C=1, solver="liblinear", penalty='l2').fit(X_train, y_train)
train_accuracy_percentage = 100 * accuracy_score(y_train, logistic_regression.predict(X_train))
val_accuracy_percentage = 100 * accuracy_score(y_val, logistic_regression.predict(X_val))

print(f"Accuracy of Logistic Regression Model for training: {train_accuracy_percentage:.2f}%")
print(f"Accuracy of Logistic Regression Model for validation: {val_accuracy_percentage:.2f}%")


# Model Evaluation using Test set
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
test_df = pd.read_csv('ExML\loan_test.csv')
test_df.head()
X_test = test_df.copy()
X_test['due_date'] = pd.to_datetime(X_test['due_date'])
X_test['effective_date'] = pd.to_datetime(X_test['effective_date'])
X_test['day_of_week'] = X_test['effective_date'].dt.dayofweek
X_test['weekend'] = X_test['day_of_week'].apply(lambda x: 1 if x > 3 else 0)
X_test['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

features = X_test[['Principal', 'terms', 'age', 'Gender', 'weekend']]
features = pd.concat([features, pd.get_dummies(X_test['education'])], axis=1)
features.drop(['Master or Above'], axis=1, inplace=True)

X_test = features.values
X_test = preprocessing.StandardScaler().fit_transform(X_test)
y_test = test_df['loan_status'].values

y_pred = Kmean.predict(X_test)

jaccard = jaccard_score(y_test, y_pred, pos_label='PAIDOFF')
f1 = f1_score(y_test, y_pred, pos_label='PAIDOFF')

print(f"Jaccard: {jaccard}")
print(f"F1-score: {f1}")

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    jaccard = jaccard_score(y, y_pred, pos_label='PAIDOFF')
    f1 = f1_score(y, y_pred, pos_label='PAIDOFF')
    return jaccard, f1

def calculate_log_loss(model, X, y):
    y_pred_proba = model.predict_proba(X)
    logloss = log_loss(y, y_pred_proba)
    return logloss

jaccard_DTree, f1_DTree = evaluate_model(decision_tree, X_test, y_test)
jaccard_SVMachine, f1_SVMachine = evaluate_model(svm_model, X_test, y_test)
jaccard_LogRegression, f1_LogRegression = evaluate_model(logistic_regression, X_test, y_test)

log_loss_LogRegression = calculate_log_loss(logistic_regression, X_test, y_test)

print(f"DTree - Jaccard: {jaccard_DTree}, F1: {f1_DTree}")
print(f"SVMachine - Jaccard: {jaccard_SVMachine}, F1: {f1_SVMachine}")
print(f"LogRegression - Jaccard: {jaccard_LogRegression}, F1: {f1_LogRegression}")
print(f"Log Loss: {log_loss_LogRegression}")




