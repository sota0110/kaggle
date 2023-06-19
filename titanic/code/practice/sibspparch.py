import pandas as pd

df = pd.read_csv("titanic/data/train.csv")

print(df.isnull().sum())

# # 兄弟・配偶者の数が多いほうが生存率が低い(0.035322)
# df_sibsp = df[['SibSp', 'Survived']]
# cor_sibsp = df_sibsp.corr()
# print(cor_sibsp)

# # 親・子供の数が多いほうが生存率が高い(0.081629)
# df_parch = df[['Parch', 'Survived']]
# cor_parch = df_parch.corr()
# print(cor_parch)

# # 兄弟・配偶者の数と親・子供の数の合計が多いほうが生存率が高い(0.016639)
# df['FamilySize'] = df['SibSp'] + df['Parch']
# df_sibsp_parch = df[['FamilySize', 'Survived']]
# cor_sibsp_parch = df_sibsp_parch.corr()
# print(cor_sibsp_parch)
