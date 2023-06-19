import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
# import xgboost as xgb
# import shap

# データの読み込み
df_train = pd.read_csv('titanic/data/train.csv')
len_train = len(df_train)
df_test = pd.read_csv('titanic/data/test.csv')
df = pd.concat([df_train, df_test], sort=False)

# ダミー変数化
df_dummy_sex = pd.get_dummies(df['Sex'])
df_dummy_embarked = pd.get_dummies(df['Embarked'])
df['Sex'] = df_dummy_sex['male']
df['Embarked'] = df_dummy_embarked['S']
df['Embarked_C'] = df_dummy_embarked['C']

# カラムの抽出
df_exctracted = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C', 'Survived']]

# 欠損値の補完
si = SimpleImputer(missing_values=np.nan, strategy='mean')
df_exctracted = si.fit_transform(df_exctracted)
df_exctracted = pd.DataFrame(df_exctracted, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C', 'Survived'])

# 説明変数と目的変数の分割
X = df_exctracted[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C']]
n_columns = len(X.columns)
y = df_exctracted['Survived']

# 標準化
X = (X - X.mean()) / X.std()

# trainとtestに分割
X = X.values
X = X.reshape(-1, n_columns)
y = y.values
X_train = X[:len_train]
y_train = y[:len_train]
X_test = X[len_train:]
y_test = y[len_train:]

# # shapの実行
# X_train_df = pd.DataFrame(X_train, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C'])
# y_train_df = pd.DataFrame(y_train, columns=['Survived'])
# model = xgb.XGBRegressor()
# model.fit(X_train_df, y_train_df)
# explainer = shap.Explainer(model)
# shap_values = explainer(X_train_df)
# shap.summary_plot(shap_values)

# 学習
reg = KNeighborsClassifier()
reg.fit(X_train, y_train)

# csvファイルに出力
df_test['Survived'] = reg.predict(X_test).astype(int)
df_result = df_test[['PassengerId', 'Survived']]
# df_result.to_csv('titanic/data/result.csv', index=False)