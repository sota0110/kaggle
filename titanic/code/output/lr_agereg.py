import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

# データの読み込み
df_train = pd.read_csv('titanic/data/train.csv')
len_train = len(df_train)
df_test = pd.read_csv('titanic/data/test.csv')
df = pd.concat([df_train, df_test], sort=False)
df = pd.DataFrame(df.values, columns=df.columns)

# 名前の処理
df['Mrs'] = 0
df.loc[df['Name'].str.contains('Mrs.'), 'Mrs'] = 1
df['Miss'] = 0
df.loc[~(df['Name'].str.contains('Mrs.')) & (df['Sex'] == 'female'), 'Miss'] = 1
df['Master'] = 0
df.loc[df['Name'].str.contains('Master.'), 'Master'] = 1

# ダミー変数化
df_dummy_sex = pd.get_dummies(df['Sex'])
df_dummy_embarked = pd.get_dummies(df['Embarked'])
df['Sex'] = df_dummy_sex['male']
df['Embarked'] = df_dummy_embarked['S']
df['Embarked_C'] = df_dummy_embarked['C']

# ageの欠損値の補完
# カラムを抽出
df_extracted_age = df[['Pclass', 'Fare', 'Embarked', 'Embarked_C', 'Mrs', 'Miss', 'Master']]

# 欠損値の補完
si = SimpleImputer(missing_values=np.nan, strategy='mean')
df_extracted_age = si.fit_transform(df_extracted_age)
df_extracted_age = pd.DataFrame(df_extracted_age, columns=['Pclass', 'Fare', 'Embarked', 'Embarked_C', 'Mrs', 'Miss', 'Master'])
df_extracted_age['Age'] = df['Age']

# trainとtestに分割
df_age_train = df_extracted_age[df_extracted_age['Age'].notnull()]
df_age_test = df_extracted_age[df_extracted_age['Age'].isnull()]

# 説明変数と目的変数の分割
X_train_age = df_age_train[['Pclass', 'Fare', 'Embarked', 'Embarked_C', 'Mrs', 'Miss', 'Master']].values
y_train_age = df_age_train['Age'].values
X_test_age = df_age_test[['Pclass', 'Fare', 'Embarked', 'Embarked_C', 'Mrs', 'Miss', 'Master']].values
n_columns_age = len(df_extracted_age.columns)-1
X_train_age = X_train_age.reshape(-1, n_columns_age)
X_test_age = X_test_age.reshape(-1, n_columns_age)

# 標準化
X_train_age = (X_train_age - X_train_age.mean()) / X_train_age.std()
X_test_age = (X_test_age - X_test_age.mean()) / X_test_age.std()

# 学習
reg = LinearRegression()
quadratic = PolynomialFeatures(degree=3)
X_quad_train = quadratic.fit_transform(X_train_age)
reg.fit(X_quad_train, y_train_age)
X_quad_test = quadratic.fit_transform(X_test_age)

# カラムの抽出
df_extracted = df[['Pclass', 'Age', 'Fare', 'Embarked', 'Embarked_C', 'Mrs', 'Miss', 'Master', 'Survived']]

# 欠損値補完
df_extracted.loc[df_extracted['Age'].isnull(), 'Age'] = reg.predict(X_quad_test)

# 欠損値の補完
df_extracted = si.fit_transform(df_extracted)
df_extracted = pd.DataFrame(df_extracted, columns=['Pclass', 'Age', 'Fare', 'Embarked', 'Embarked_C', 'Mrs', 'Miss', 'Master', 'Survived'])

# 説明変数と目的変数の分割
X = df_extracted[['Pclass', 'Age', 'Fare', 'Embarked', 'Embarked_C', 'Mrs', 'Miss', 'Master']]
n_columns = len(X.columns)
y = df_extracted['Survived']

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

# 学習
reg = LogisticRegression()
reg.fit(X_train, y_train)

# csvファイルに出力
df_test['Survived'] = reg.predict(X_test).astype(int)
df_result = df_test[['PassengerId', 'Survived']]
df_result.to_csv('titanic/data/result_lr_agereg.csv', index=False)