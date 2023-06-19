import pandas as pd

df = pd.read_csv('titanic/data/train.csv')

# 男性よりも女性の方が生存率が高い(0.543351)
df_sex = pd.get_dummies(df['Sex'])
df_sex['Survived'] = df['Survived']
df_sex = df_sex[['male', 'Survived']]
cor_sex = df_sex.corr()
print(cor_sex)

# チケットクラスが高いほうが生存率が高い(0.338481) 
df_pclass = df[['Pclass', 'Survived']]
cor_pclass = df_pclass.corr()
print(cor_pclass)

# 兄弟・配偶者の数が多いほうが生存率が低い(0.035322)
df_sibsp = df[['SibSp', 'Survived']]
cor_sibsp = df_sibsp.corr()
print(cor_sibsp)

# 親・子供の数が多いほうが生存率が高い(0.081629)
df_parch = df[['Parch', 'Survived']]
cor_parch = df_parch.corr()
print(cor_parch)

# 運賃が高いほうが生存率が高い(0.257307)
df_fare = df[['Fare', 'Survived']]
cor_fare = df_fare.corr()
print(cor_fare)

# 乗船港がCherbourgの方が生存率が高い(0.16824)
# 乗船港がQueenstownの方が生存率が高い(0.00365)
# 乗船港がSouthamptonの方が生存率が低い(-0.15566)
df_embarked = pd.get_dummies(df['Embarked'])
df_embarked['Survived'] = df['Survived']
df_embarked = df_embarked[['C', 'Q', 'S', 'Survived']]
cor_embarked = df_embarked.corr()
print(cor_embarked)

# titleがMrの方が生存率が低い(-0.566512)
# titleがMissの方が生存率が高い(0.341325)
# titleがMrsの方が生存率が高い(0.344223)
# titleがMasterの方が生存率が高い(0.085221)
df_title = df.copy()
df_title[["Mr", "Miss", "Mrs", "Master"]] = 0
df_title = df_title[["Name", "Sex", "Mr", "Miss", "Mrs", "Master", "Survived"]]
df_title.loc[df_title.Name.str.contains("Mrs."), "Mrs"] = 1
df_title.loc[~(df_title.Name.str.contains("Mrs.")) & (df_title.Sex == "female"), "Miss"] = 1
df_title.loc[df_title.Name.str.contains("Master."), "Master"] = 1
df_title.loc[~(df_title.Name.str.contains("Master.")) & (df_title.Sex == "male"), "Mr"] = 1
cor_title = df_title.corr()
print(cor_title["Survived"])

# Ticketの整理
df_ticket = df.copy()
df_ticket.Ticket = df_ticket.Ticket.str[:1]
# 数字以外の文字列を0に置き換える
df_ticket.Ticket = df_ticket.Ticket.str.replace("[^0-9]", "0")
df_ticket.Ticket = df_ticket.Ticket.astype(int)
# print(df_ticket.Ticket.str[:1].value_counts())
cor_ticket = df_ticket.corr()
print(cor_ticket)