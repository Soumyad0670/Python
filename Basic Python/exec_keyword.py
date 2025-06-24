# Using exec with globals and locals
global_scope = {"x": 5}
local_scope = {}

exec("y = x + 10", global_scope, local_scope)
print(local_scope["y"])  # Output: 15

a = (1, 45, 342, 3424, False, 45, 45, "rohan", "Shivam")
print(type(a))
print(a)
no = a.count(45)
i = a.index(3424)
print(no)
print(i)
print(len(a))

# b = (3, 34, 44, "Harry")
# a[2] = "Larry"  # Tuples are immutable

# Using enumerate with a tuple
for index, value in enumerate(a):
    print(index, value)

# Creating a tuple
my_tuple = (1, 2, 3, 4, 5, 3)

# Accessing elements
print(my_tuple[0])  # Output: 1

# Counting elements
count = my_tuple.count(3)
print(count)  # Output: 2

# Finding index
index = my_tuple.index(4)
print(index)  # Output: 3

# Length of the tuple
length = len(my_tuple)
print(length)  # Output: 6

# Using exec to execute a string of code
code = """
for i in range(3):
    print(f"Exec loop: {i}")
"""
exec(code)
