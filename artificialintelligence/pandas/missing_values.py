import pandas as pd
data=pd.read_csv('/home/ila/PycharmProjects/pythonworks/artificialintelligence/pandas/files/missingvalues.txt')
print(data.columns)
data.columns=data.columns.z_str.replace(" ", "")
print("after removing space in columns",data.columns)
print(data.tail())
print(data.isnull().tail())

#no of missing values
print("total missing values",data.isnull().sum())#default is axis=0
print("total missing values",data.isnull().sum(axis=1))

#creating series
new=pd.Series([True,True,False,True])
print(type(new))
print(new)
print(new.sum())#only true will be taken and printed

print(data.shape)
#drop a row if any of its avlue is missing
# data.dropna(how='any',inplace=True)
print(data.dropna(how='any').shape)
print(data.dropna(how='all').shape)#if all the values of a row are missing
print(data.shape)
print(data.dropna(subset=['ColorsReported', 'ShapeReported'], how='all').shape)
print(data.ShapeReported.value_counts())
print(data.ShapeReported.value_counts(dropna=False))
print(data.fillna(value='ila'))#can also used to fill an particular series



