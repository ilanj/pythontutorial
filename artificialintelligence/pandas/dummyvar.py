'''
when dummies are used, for example there are 2 values in sex series, m and f, then original sex series will be replaced by sex_m and sex_f
and it will contain binary values 0 and 1 only. for the values present 1 will be there. (Feature Encoding)
'''
import pandas as pd
data=pd.read_csv("files/kaggletrain.txt")
x=data.iloc[:,0:2]
y=data.iloc[:,-1]
data['gender_binary']=data.Sex.map({'female':0,'male':1})
print(data.head())
print(pd.get_dummies(data.Sex).head())
print(pd.get_dummies(data.Sex).iloc[:,1:].head())
print(pd.get_dummies(data.Sex,prefix='Sex_').iloc[:,1:].head())
print(type(pd.get_dummies(data.Sex)))
print(data.Pclass.value_counts())

print(pd.get_dummies(data.Pclass,prefix='Pclass_').iloc[:,:].head())
temp=pd.get_dummies(data.Pclass,prefix='Pclass_').iloc[:,:]
# data=pd.concat([data,temp],axis=1)
print(data.describe)#temp dataframe is appended to data

print(pd.get_dummies(data,columns=['Sex','Pclass']))
print(pd.get_dummies(data,columns=['Sex','Pclass'],drop_first=True))#Sex_female and embarked_1 will be dropped






