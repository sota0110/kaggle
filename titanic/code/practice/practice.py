import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

df = pd.read_csv('titanic/data/train.csv')
df_dummy_sex = pd.get_dummies(df['Sex'])
df_dummy_embarked = pd.get_dummies(df['Embarked'])
df['Sex'] = df_dummy_sex['male']
df['Embarked'] = df_dummy_embarked['S']
df['Embarked_C'] = df_dummy_embarked['C']

df_exctracted = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C', 'Survived']].values
si = SimpleImputer(missing_values=np.nan, strategy='mean')

df_exctracted = si.fit_transform(df_exctracted)
df_exctracted = pd.DataFrame(df_exctracted, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C', 'Survived'])

X = df_exctracted[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C']]
X = (X - X.mean()) / X.std()
n_columns = len(X.columns)
y = df_exctracted['Survived']
n = len(df_exctracted)
n_train = int(n * 0.7)

X_train = X[:n_train].values
X_train = X_train.reshape(-1, n_columns)
y_train = y[:n_train].values
X_test = X[n_train:].values
X_test = X_test.reshape(-1, n_columns)
y_test = y[n_train:].values

reg = KNeighborsClassifier()
reg.fit(X_train, y_train)
print(confusion_matrix(y_test, reg.predict(X_test)))
print("正解率＝", accuracy_score(y_test, reg.predict(X_test)))
