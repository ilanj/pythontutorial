import pandas as pd
col_names=['sl_no','age','gender','role','id']
data=pd.read_table("/home/ila/PycharmProjects/pythonworks/artificialintelligence/pandas/files/ex.txt",sep="|",names=col_names)
print(data.dtypes)
data.age=data.age.astype(float)#no need inplace parameter
print(data.dtypes)

data1=pd.read_table("/home/ila/PycharmProjects/pythonworks/artificialintelligence/pandas/files/ex.txt",sep="|",names=col_names,dtype={'sl_no':float,'age':float})
print(data1.dtypes)


print(data.dtypes)
# print(data.id.str.replace('\D',3).astype(float).mean())
data.id=data.id.z_str.replace("\D", '3')
print(data.id)#now everything is converted to digits
data.id=data.id.astype(float)
print(data.dtypes)
