import pandas as pd
col_names=['sl_no','age','gender','role','id']
data=pd.read_table("/home/ila/PycharmProjects/pythonworks/artificialintelligence/pandas/files/ex.txt",sep="|",names=col_names)
print(data.head())
print(data['role'].z_str.upper())
print(data.role.z_str.lower())
print(data.role.z_str.contains('or'))#will only display boolean result
print(data[data.role.z_str.contains('or')])#will display as a table(roles educator and administrator will be displayed)

