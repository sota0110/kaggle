import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df_train = pd.read_csv("titanic/data/train.csv")
df_test = pd.read_csv("titanic/data/test.csv")
df = pd.concat([df_train, df_test], sort=False)
df = pd.DataFrame(df.values, columns=df.columns)

# カラムの抽出.fillna(df.Fare.median())
df.Embarked = df.Embarked.fillna("S")
# print(df.isnull().sum())

# Ticketの整理
df.Ticket = df.Ticket.str[:1]
sur = len(df[(df.Ticket == '1') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '1') & (df.Survived == 0)])
print('1で生き残る：', sur)
print('1で生き残らない：', dea)
print('1で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '2') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '2') & (df.Survived == 0)])
print('2で生き残る：', sur)
print('2で生き残らない：', dea)
print('2で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '3') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '3') & (df.Survived == 0)])
print('3で生き残る：', sur)
print('3で生き残らない：', dea)
print('3で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '4') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '4') & (df.Survived == 0)])
print('4で生き残る：', sur)
print('4で生き残らない：', dea)
print('4で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '5') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '5') & (df.Survived == 0)])
print('5で生き残る：', sur)
print('5で生き残らない：', dea)
print('5で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '6') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '6') & (df.Survived == 0)])
print('6で生き残る：', sur)
print('6で生き残らない：', dea)
print('6で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '7') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '7') & (df.Survived == 0)])
print('7で生き残る：', sur)
print('7で生き残らない：', dea)
print('7で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '8') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '8') & (df.Survived == 0)])
print('8で生き残る：', sur)
print('8で生き残らない：', dea)
print('8で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == '9') & (df.Survived == 1)])
dea = len(df[(df.Ticket == '9') & (df.Survived == 0)])
print('9で生き残る：', sur)
print('9で生き残らない：', dea)
print('9で生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == 'A') & (df.Survived == 1)])
dea = len(df[(df.Ticket == 'A') & (df.Survived == 0)])
print('Aで生き残る：', sur)
print('Aで生き残らない：', dea)
print('Aで生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == 'C') & (df.Survived == 1)])
dea = len(df[(df.Ticket == 'C') & (df.Survived == 0)])
print('Cで生き残る：', sur)
print('Cで生き残らない：', dea)
print('Cで生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == 'F') & (df.Survived == 1)])
dea = len(df[(df.Ticket == 'F') & (df.Survived == 0)])
print('Fで生き残る：', sur)
print('Fで生き残らない：', dea)
print('Fで生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == 'L') & (df.Survived == 1)])
dea = len(df[(df.Ticket == 'L') & (df.Survived == 0)])
print('Lで生き残る：', sur)
print('Lで生き残らない：', dea)
print('Lで生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == 'P') & (df.Survived == 1)])
dea = len(df[(df.Ticket == 'P') & (df.Survived == 0)])
print('Pで生き残る：', sur)
print('Pで生き残らない：', dea)
print('Pで生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == 'S') & (df.Survived == 1)])
dea = len(df[(df.Ticket == 'S') & (df.Survived == 0)])
print('Sで生き残る：', sur)
print('Sで生き残らない：', dea)
print('Sで生き残る割合：', sur/(sur+dea))
sur = len(df[(df.Ticket == 'W') & (df.Survived == 1)])
dea = len(df[(df.Ticket == 'W') & (df.Survived == 0)])
print('Wで生き残る：', sur)
print('Wで生き残らない：', dea)
print('Wで生き残る割合：', sur/(sur+dea))