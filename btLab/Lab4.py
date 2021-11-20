import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def load_data(filename):
    return pd.read_csv(filename)
filename=r'H:\NhapMonHocSau\HocSau\titanic_disaster.csv'
df=load_data(filename)
#câu 1
print(df.head(10))

#câu2
sns.heatmap(df.isna(),cmap='viridis')
print(plt.show())
df_thieu_age=df[df['Age'].isna()]
print(df_thieu_age.count())
df_thieu_carbin=df[df['Cabin'].isna()]
print(df_thieu_carbin.count())
df_thieu_Embarked=df[df['Embarked'].isna()]
print(df_thieu_Embarked.count())
# Nhận xét : dữ liệu age có 177 dòng null, có 687 dòng cabin=null , có 
#câu 3
df[['firstname','secondname']]=df['Name'].str.split(',',expand=True)
df=df.drop('Name',axis=1)

#câu 4
df['Sex'].replace('male','M',inplace=True)
df['Sex'].replace('female','F',inplace=True)

#câu5
sns.boxplot(y='Age',x='Pclass',data=df)
print(plt.show())

# thay dữ liệu age bị thiếu
df['Age'][df['Pclass']==1]=df['Age'][df['Pclass']==1].fillna(df['Age'][df['Pclass']==1].mean())
df['Age'][df['Pclass']==2]=df['Age'][df['Pclass']==2].fillna(df['Age'][df['Pclass']==2].mean())
df['Age'][df['Pclass']==3]=df['Age'][df['Pclass']==3].fillna(df['Age'][df['Pclass']==3].mean())

sns.heatmap(df.isna(),yticklabels=False,cbar=True,cmap='viridis')
print(plt.show())

#câu 6
df.loc[df['Age']<=12,'Agegroup']='Kid'
df.loc[(df['Age']>12)&(df['Age']>=18),'Agegroup']='Adult'
df.loc[df['Age']>60,'Agegroup']='Older'

#câu 7

#câu 8
df['familySize']= df['SibSp'] +df['Parch'] +1

#câu 9
df.loc[df['familySize']==0,'Alone']=1
df.loc[df['familySize']!=0,'Alone']=0

#câu 12
df.groupby(['Survived','Sex'])['Sex'].agg(['count']).plot.bar()
print(plt.show())

#Các kết quả trong phần img