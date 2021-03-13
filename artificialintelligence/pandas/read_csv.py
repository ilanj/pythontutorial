import pandas as pd
# Zip Code	Total Population	Median Age	Total Males	Total Females	Total Households	Average Household Size

data=pd.read_csv("files/2010_Census_Populations_by_Zip_Code.csv")
print(data["Zip Code"])
print(type(data))
# print(type(data["Zip Code"]))
# print(data.Zip Code) #can be used in both ways
# print(data.head())
print(data.shape)#rows,columns
print(data.head) #prints data and with cols and rows as metadata
print(data.head())
print(data["Zip Code"]+data["Total Population"])#fetch 2 columns and add
#if u want to add this as a new column
data['ziopulation'] = data['Median Age']+data['Total Population'].apply(int)
temp=data['ziopulation']
print(data.head())
print(data['ziopulation'])
print(data.describe())#gives count mean std min 25% 50% 75% and max for numeric columns
print("shape",data.shape)
print("dtypes",data.dtypes)
#to remove spaces in column names
data.columns=data.columns.z_str.replace(" ", "")
print(data.columns)
print("after removing space",)
# # print("describe int",data.describe(include=['int64']))#take type from previous using dtypes
# print("columns->",data.columns)
# print("rename->",data.rename(columns={'Zip Code':'Zip_Code','Total Population':'Total_Population'}))
# print("after rename",data.dtypes)#will not reflect here. if this should afftect original add 'inplace' parameter
# print("rename->",data.rename(columns={'Zip Code':'Zip_Code','Total Population':'Total_Population'},inplace=True))
# print("after rename with inplace=true",data.dtypes)#now rename reflected
# newcolnames=['zipcode','totalpopulation','medianage','totalmales','totalfemales','totalhouseholds','avghouseholdsize','zipandpopulation']
# data.columns=newcolnames
# print("after renaming fully",data.dtypes)#now renamed fully
# '''while reading file itself can rename columns with data=pd.read_csv("G:\workspace\pythontutorial\iofiles//2010_Census_Populations_by_Zip_Code.csv",names=newcolnames,header=0)
#
# '''
#
#
#
