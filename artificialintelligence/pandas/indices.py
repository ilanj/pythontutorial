import pandas as pd
import numpy as np
col_names=['sl_no','age','gender','role','id']
data=pd.read_table("files/ex.txt",sep="|",header=None)
print(data.columns)
print(data.index)
print(data.head())#both rows and column are characterized by indexes
#index represent each row from 0 to n. when there is noe column name, index will be the default
print(data.loc[25,3])#row 25 , 3 rd column. can also give column names
data1=pd.read_table("files/ex.txt",sep="|",names=col_names)
print(data1.loc[25,'age'])#row 25 , 3 rd column. can also give column names

data1.set_index('id',inplace=True)
print(data1.head())
print(data1.index)

data1.set_index('role',inplace=True)
print(data1.head())
print(data1.index)
print(data1.head())#u can c a row name role is empty
data1.index.name=None
print(data1.head())#the above menioned empty row is removed
print(data1.tail())
print(data1.loc[data1.age==18,'gender'])
data1.index.name='i_role'
print(data1.head())
print(data1.reset_index(inplace=True))
print(data1.tail())
print(data1.describe().index)
print(data1.describe().columns)
print(data1.describe().loc['25%','age'])

print(data1.age.value_counts().values)#will describe the frequency
print(data1.age.value_counts())
print(data1.i_role.value_counts())
print(data1.i_role.value_counts()['student'])

print(data1.i_role.value_counts().sort_values())#ascending
print(data1.i_role.value_counts().sort_index())#ascending
print(data1.head())
data1.set_index('i_role',inplace=True)
print(data1.head())

#creating a new series
new_role=pd.Series([25,30],index=['student','technician'],name='percentage_new')
new_role1=pd.Series([785,774],index=[2.0,3.0],name='percentage_new1')
print("type of new role",type(new_role))
print(new_role)

temp=data1.sl_no*new_role#debug and view temp

data1.set_index('sl_no',inplace=True)

data2=pd.concat([data1,new_role1])
print(data2.head())
print()
# print(data1.sl_no*new_role)
# print(type(data1.sl_no*new_role.head()))
# pd.concat([data1,new_role],axis=1)
# print()
# # print(data1.tail())
#
#
