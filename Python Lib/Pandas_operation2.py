import pandas as s
a=s.read_csv("C:/Users/KIIT/Documents/emp_data.csv")
print(a)
print(a['City']=='Noida')
grp=a.groupby('City')
print(grp['Salary'].get_group('Banglore').sum().mean())
print(a.shape)
print(a.columns)
print(a.dtypes)
print(a.describe())
print(a.info())
print(a.head())
print(a.tail())
print(a.pivot_table(values='Salary',index='City',aggfunc='mean'))
print(a.loc[3:7])
print(a.iloc[5:7])
city_map = {
    'Banglore': 'Bangalore',
    'Delhi': 'New Delhi',
    'Mumbai': 'Bombay'
}
a['City'] = a['City'].map(city_map)
print("\nUpdated DataFrame:")
print(a)