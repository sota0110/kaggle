import pandas as pd

df = pd.read_csv('titanic/data/train.csv')

df['Mrs'] = 0
df.loc[df['Name'].str.contains('Mrs.'), 'Mrs'] = 1
df['Miss'] = 0
df.loc[~(df['Name'].str.contains('Mrs.')) & (df['Sex'] == 'female'), 'Miss'] = 1
df['Master'] = 0
df.loc[df['Name'].str.contains('Master.'), 'Master'] = 1
df['Mr'] = 0
df.loc[~(df['Name'].str.contains('Master.')) & (df['Sex'] == 'male'), 'Mr'] = 1

df_name = df[['Mrs', 'Miss', 'Master', 'Mr', 'Survived']]
cor_name = df_name.corr()
print(cor_name)