# EDA- exploratory data analysis

import pandas as pd

# Series

lst=[1,2,3,4,5]
print(lst)
print(pd.Series([1,2,3,4,5],index=['a','b','c','d','e'],name='Ratings'))

#DataFrame
# using list or dictionary

lst=[[1,2],[2123,23],[3212321,313]]
print(lst[2][0])

print(pd.DataFrame(lst,columns=['a','b']))

group={
    'Fruits':['apple','mango','arrange'],
    'Quantity':[6,5,12],
    'Like/Dislike':['l','d','l']
}
df=pd.DataFrame(group)
print(df)

print(group['Fruits'])
print(group['Quantity'])
print(df.ndim)
print(df.shape)

group['Price']=[50,34,24]
print(pd.DataFrame(group))
print(group['Price'])

# Arithematic operations

dic=pd.DataFrame({'A':[1,2,3,4,5], 'B':[2,3,4,5,6]})
dic['C']=dic['A']+dic['B']
dic['D']=dic['C']*dic['B']
print(pd.DataFrame(dic.insert(0,'Name',['a','b','c','d','e'])))
print(dic)

# Reading the CSV file
df = pd.read_csv("C:/Users/KIIT/Documents/petrol_consumption.csv")
print(df)
print(df.shape)
print(df.describe())
# Head and Tail
print(df.head())
print(df.tail())

# Indexing and Slicing
print(df.columns)

#data types
print(df.dtypes)

# information about the data
print(df.info())

# Null values
print(df.isna().sum())  
print(df.isna().mean()*100)
print(df.isnull())

# Filling the null values  with 0
print(df.fillna(0))

print(df)

# Dropping the null values
print(df.dropna())

#Duplicate data
print(df.duplicated().sum())
print(df[df.duplicated])

print(df.nunique())
print(df['Petrol_tax'].unique())
print(df['Petrol_tax'].value_counts())
print(df['Petrol_tax'].mean())
print(df['Petrol_tax'].median())
print(df['Petrol_tax'].mode())
print(df['Petrol_tax'].std())
print(df['Petrol_tax'].var())

d=pd.read_csv("C:/Users/KIIT/Documents/emp_data.csv")
print(d)
print(d[d['City']=='Banglore'])
print(d.head())
print(d.tail())
print(d['City'].value_counts())
group=d.groupby('City')
print(group.get_group('Banglore'))
print(group.get_group('Banglore').min())
print(group.get_group('Banglore').max())
print("Mean Salary of employees working in Banglore",group['Salary'].get_group('Banglore').sum().mean())
print("Median Salary of employees working in Banglore",group['Salary'].get_group('Gurgaon').median())
print("Mode of Salary of employees working in Gurgaon\n",group['Salary'].get_group('Gurgaon').mode())
print("Standard Deviation of Salary of employees working in Pune",group['Salary'].get_group('Pune').std())
print("variance of Salary of employees working in Pune",group['Salary'].get_group('Pune').var())
print(d.set_index('Emp_id'))
print(d.reset_index(inplace=True))
