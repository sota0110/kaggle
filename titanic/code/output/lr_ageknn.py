import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# データの読み込み
df_train = pd.read_csv("titanic/data/train.csv")
len_train = len(df_train)
df_test = pd.read_csv("titanic/data/test.csv")
df = pd.concat([df_train, df_test], sort=False)
df = pd.DataFrame(df.values, columns=df.columns)

# NameからTitleを作成
df.loc[df.Name.str.contains("Mrs."), "Title"] = 0
df.loc[~(df.Name.str.contains("Mrs.")) & (df.Sex == "female"), "Title"] = 1
df.loc[df.Name.str.contains("Master."), "Title"] = 2
df.loc[~(df.Name.str.contains("Master.")) & (df.Sex == "male"), "Title"] = 3

# ダミー変数化
df.loc[df.Sex == "male", "Sex"] = 0
df.loc[df.Sex == "female", "Sex"] = 1
df.loc[df.Embarked == "C", "Embarked"] = 0
df.loc[df.Embarked == "Q", "Embarked"] = 1
df.loc[(df.Embarked != 0) & (df.Embarked != 1), "Embarked"] = 2

# ageの欠損値の補完
# カラムを抽出
columns = ["Pclass", "Fare", "Embarked", "Title"]
df_age = df[columns]

# 欠損値の補完
df_age.loc[df.Fare.isnull(), "Fare"] = df_age.Fare.median()
df_age = df_age.assign(Age=df.Age)

# trainとtestに分割
df_age_train = df_age[df_age.Age.notnull()]
df_age_test = df_age[df_age.Age.isnull()]

# 説明変数と目的変数の分割
X_train_age = df_age_train[columns].values
y_train_age = df_age_train.Age.values
X_test_age = df_age_test[columns].values
n_columns = len(df_age.columns) - 1
X_train_age = X_train_age.reshape(-1, n_columns)
X_test_age = X_test_age.reshape(-1, n_columns)
y_train_age = y_train_age.astype("int")

# 標準化
X_train_age = (X_train_age - X_train_age.mean()) / X_train_age.std()
X_test_age = (X_test_age - X_test_age.mean()) / X_test_age.std()

# 学習
reg = KNeighborsClassifier(n_neighbors=3)
reg.fit(X_train_age, y_train_age)

# 欠損値補完
df_age.loc[df_age.Age.isnull(), "Age"] = reg.predict(X_test_age)

# カラムの抽出
columns.append("Age")
columns_sur = columns.copy()
columns_sur.append("Survived")
df_extracted = df_age
df_extracted = df_extracted.assign(Survived=df.Survived)

# 説明変数と目的変数の分割
X = df_extracted[columns]
n_columns = len(X.columns)
y = df_extracted.Survived

# 標準化
X = (X - X.mean()) / X.std()

# trainとtestに分割
X = X.values
X = X.reshape(-1, n_columns)
y = y.values
X_train = X[:len_train]
y_train = y[:len_train].astype(int)
X_test = X[len_train:]
y_test = y[len_train:]

# 学習
reg = LogisticRegression()
reg.fit(X_train, y_train)

# csvファイルに出力
df_test["Survived"] = reg.predict(X_test).astype(int)
df_result = df_test[["PassengerId", "Survived"]]
df_result.to_csv("titanic/data/result_lr_ageknn.csv", index=False)