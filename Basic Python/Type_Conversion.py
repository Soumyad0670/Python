# Python code to demonstrate Type conversion
# using int(), float(), str(), list(), and tuple()

# initializing string
s = "10110000"

# printing string converting to int base 2
c = int(s, 2)
print("After converting to integer base 2: ")
print(c)

# printing string converting to float
e = float(s)
print("After converting to float: ")
print(e)

# initializing integer
i = 123

# printing integer converting to string
s = str(i)
print("After converting to string: ")
print(s)

# initializing tuple
t = (1, 2, 3, 4)

# printing tuple converting to list
l = list(t)
print("After converting to list: ")
print(l)

# initializing list
l = [1, 2, 3, 4]

# printing list converting to tuple
t = tuple(l)
print("After converting to tuple: ")
print(t)