#list comprehension
lst= list(range(1,11))
for i in lst:
    even=[]
    if i % 2 == 0:
        even.append(i)
        print(even)

# comprehension is done to reduce time and space complexity
#list comprehension
print([i for i in range(1,11) if i % 2 == 0])

print([i for i in range(1,11) if i % 2 != 0])

# Creating a dictionary with dictionary comprehension
squares = {x: x**2 for x in range(1, 6)} 
print(squares)  # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Creating a dictionary with a condition
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)  # Output: {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}# List comprehension to create a list of odd numbers

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
odd = [i for i in lst if i % 2 != 0]
print(odd)  # Output: [1, 3, 5, 7, 9]

# Dictionary comprehension to create a dictionary with squares of numbers
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Dictionary comprehension with a condition to create a dictionary with even squares
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)  # Output: {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

# Example dictionary
d = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
l=[1,2,3,4]

# Dictionary comprehension to create a dictionary with squared values
squared_values = {key: value**2 for key, value in l.items()}
print(squared_values)  # Output: {'1': 1, '2': 4, '3': 9, '4': 16, '5': 25}

# Dictionary comprehension with a condition to filter even values
even_values = {key: value for key, value in d.items() if value % 2 == 0}
print(even_values)  # Output: {'2': 2, '4': 4}

d={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
new_d={x:y**4 for x,y in d.items()}
even_d={x:y**4 for x,y in d.items() if y%2==0}
odd_d={x:y**4 for x,y in d.items() if y%2!=0}
print(new_d)
print(even_d)
print(odd_d)

# List of elements
lst = ['a', 'b', 'c', 'd']
# Creating a dictionary with list elements as keys and their indices as values
dict_from_list = {value: index for index, value in enumerate(lst)}
print(dict_from_list)  # Output: {'a': 0, 'b': 1, 'c': 2, 'd': 3}# List of elements

lst = [1, 2, 3, 4]
# Creating a dictionary with list elements as keys and their squares as values
squared_values = {value: value**2 for value in lst}
print(squared_values)  # Output: {1: 1, 2: 4, 3: 9, 4: 16}# List of tuples

lst_of_tuples = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
# Creating a dictionary from a list of tuples
dict_from_tuples = dict(lst_of_tuples)
print(dict_from_tuples)  # Output: {'a': 1, 'b': 2, 'c': 3, 'd': 4}

