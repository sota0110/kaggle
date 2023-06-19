import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("titanic/data/train.csv")
df_test = pd.read_csv("titanic/data/test.csv")
df = pd.concat([df_train, df_test], sort=False)
df = pd.DataFrame(df.values, columns=df.columns)

# 名前の処理
df["Mrs"] = 0
df.loc[df["Name"].str.contains("Mrs."), "Mrs"] = 1
df["Miss"] = 0
df.loc[~(df["Name"].str.contains("Mrs.")) & (df["Sex"] == "female"), "Miss"] = 1
df["Master"] = 0
df.loc[df["Name"].str.contains("Master."), "Master"] = 1

# # ダミー変数化
# df_dummy_sex = pd.get_dummies(df["Sex"])
# df_dummy_embarked = pd.get_dummies(df["Embarked"])
# df["Sex"] = df_dummy_sex["male"]
# df["Embarked"] = df_dummy_embarked["S"]
# df["Embarked_C"] = df_dummy_embarked["C"]

# 欠損値の補完
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# カラムの抽出
df_extracted_age = df[["Pclass", "Fare", "Embarked", "Mrs", "Miss", "Master", "Age"]]

# 欠損値の補完
df_extracted_age.loc[df_extracted_age["Embarked"].isnull(), "Embarked"] = 0

# trainとtestに分割
df_age_train = df_extracted_age[df_extracted_age["Age"].notnull()]
df_age_test = df_extracted_age[df_extracted_age["Age"].isnull()]

# ageとfareのグラフ
df_graph = df_age_train[["Age", "Pclass", "Fare", "Embarked"]]
sns.pairplot(df_graph, hue="Embarked")
plt.show()

# # 説明変数と目的変数の分割
# X_train_age = df_age_train[
#     ["Pclass", "Fare", "Embarked", "Embarked_C", "Mrs", "Miss", "Master"]
# ].values
# y_train_age = df_age_train["Age"].values
# X_test_age = df_age_test[
#     ["Pclass", "Fare", "Embarked", "Embarked_C", "Mrs", "Miss", "Master"]
# ].values
# n_columns_age = len(df_extracted_age.columns) - 1
# X_train_age = X_train_age.reshape(-1, n_columns_age)
# X_test_age = X_test_age.reshape(-1, n_columns_age)
# y_train_age = y_train_age.astype("int")

# # 標準化
# X_train_age = (X_train_age - X_train_age.mean()) / X_train_age.std()
# X_test_age = (X_test_age - X_test_age.mean()) / X_test_age.std()

# # 学習
# reg = KNeighborsClassifier(n_neighbors=3)
# reg.fit(X_train_age, y_train_age)

# # 結果を出力
# y_test_pred = reg.predict(X_test_age).astype(int)
# print(y_test_pred)
