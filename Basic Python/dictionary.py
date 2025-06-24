marks={"Harry":100,"Subham":56,"Rohan":23}#--->dictionary are mutable
# marks.__subclasshook__
print(marks.items())
print(marks.keys())
print(marks.values())

# dictionary is mutable.
# key in a dictionary is unique and immuatable but the values will be mutable.
# therefore we can update the dictionary.

marks.update({"Harry":99, "Renuka":100})
print(marks)
print(marks.get("Harry"))
print(marks["Harry"])

# Creating a dictionary
my_dict = {"name": "Alice", "age": 25, "city": "New York"}

# Accessing values
print(my_dict["name"])  # Output: Alice

# Adding or updating key-value pairs
my_dict["email"] = "alice@example.com"
my_dict["age"] = 26
print(my_dict)

# Removing key-value pairs
my_dict.pop("city")
print(my_dict)

# Getting a value
age = my_dict.get("age")
print(age)
email= my_dict.get("email")
print(email)

# Checking for keys
has_city = "city" in my_dict
print(has_city)

# Getting keys, values, and items
keys = my_dict.keys()
values = my_dict.values()
items = my_dict.items()
print(keys)
print(values)
print(items)

# Clearing the dictionary
my_dict.clear()
print(my_dict)

# Copying the dictionary
new_dict = my_dict.copy()
print(new_dict)


import time
import pandas as pd

# Creating a dictionary
data = {
    'Name': ['Sourav', 'Rahul', 'Soumya'],
    'Age': [24, 25, 26],
    'Salary': ['67k', '70k', '75k']
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Displaying the DataFrame
print(df)

import _pydatetime as datetime
 
# Creating a dictionary
date={
    'day': 18,
    'month': 1,
    'year': 2025
}
hour={
    'hour': 10,
    'minute': 30,
    'second': 45
}

dt=datetime.datetime(date['year'], date['month'], date['day'], hour['hour'], hour['minute'], hour['second'])
print(dt)


import numpy as np

# Creating a dictionary
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Converting the DataFrame to a NumPy array
array = df.to_numpy()

# Displaying the NumPy array
print(array)

import pyflakes as flake

# Creating a dictionary
data = {
    'Name':['Soumya'],
    'Age':[23],
    'City':["Kolkata"]
}

# Creating a DataFrame

df = pd.DataFrame(data)

# Displaying the DataFrame
print(df)