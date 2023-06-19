import pandas as pd

df = pd.read_csv("titanic/data/train.csv")

df_train = df[df["Cabin"].notnull()]
df_test = df[df["Cabin"].isnull()]

# df.loc[df['Cabin'].notnull(), 'Cabin'] = 1
# df.loc[df['Cabin'].isnull(), 'Cabin'] = 0
# print('Cabinがあって生き残る：', len(df[(df['Cabin'] == 1) & (df['Survived'] == 1)]))
# print('Cabinがあって生き残らない：', len(df[(df['Cabin'] == 1) & (df['Survived'] == 0)]))
# print('Cabinがなくて生き残る：', len(df[(df['Cabin'] == 0) & (df['Survived'] == 1)]))
# print('Cabinがなくて生き残らない：', len(df[(df['Cabin'] == 0) & (df['Survived'] == 0)]))

df_cabin = df_train["Cabin"].str[:1]
df_cabin = pd.get_dummies(df_cabin)
df_cabin["Survived"] = df_train["Survived"]
# print('Aで生き残る：', len(df_cabin[(df_cabin['A'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Aで生き残らない：', len(df_cabin[(df_cabin['A'] == 1) & (df_cabin['Survived'] == 0)]))
# print('Bで生き残る：', len(df_cabin[(df_cabin['B'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Bで生き残らない：', len(df_cabin[(df_cabin['B'] == 1) & (df_cabin['Survived'] == 0)]))
# print('Cで生き残る：', len(df_cabin[(df_cabin['C'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Cで生き残らない：', len(df_cabin[(df_cabin['C'] == 1) & (df_cabin['Survived'] == 0)]))
# print('Dで生き残る：', len(df_cabin[(df_cabin['D'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Dで生き残らない：', len(df_cabin[(df_cabin['D'] == 1) & (df_cabin['Survived'] == 0)]))
# print('Eで生き残る：', len(df_cabin[(df_cabin['E'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Eで生き残らない：', len(df_cabin[(df_cabin['E'] == 1) & (df_cabin['Survived'] == 0)]))
# print('Fで生き残る：', len(df_cabin[(df_cabin['F'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Fで生き残らない：', len(df_cabin[(df_cabin['F'] == 1) & (df_cabin['Survived'] == 0)]))
# print('Gで生き残る：', len(df_cabin[(df_cabin['G'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Gで生き残らない：', len(df_cabin[(df_cabin['G'] == 1) & (df_cabin['Survived'] == 0)]))
# print('Tで生き残る：', len(df_cabin[(df_cabin['T'] == 1) & (df_cabin['Survived'] == 1)]))
# print('Tで生き残らない：', len(df_cabin[(df_cabin['T'] == 1) & (df_cabin['Survived'] == 0)]))

df = pd.read_csv("titanic/data/train.csv")
df["Cabin"] = df["Cabin"].str[:1]
df["Cabin"] = df["Cabin"].fillna(1)
df.loc[df["Cabin"] == "A", "Cabin"] = 2
df.loc[df["Cabin"] == "B", "Cabin"] = 8
df.loc[df["Cabin"] == "C", "Cabin"] = 4
df.loc[df["Cabin"] == "D", "Cabin"] = 7
df.loc[df["Cabin"] == "E", "Cabin"] = 6
df.loc[df["Cabin"] == "F", "Cabin"] = 5
df.loc[df["Cabin"] == "G", "Cabin"] = 3
df.loc[df["Cabin"] == "T", "Cabin"] = 0
print(df["Cabin"].value_counts())
