age=21
if (age>=18):
    print("i am eligible")
else:
    print("not eligible")

def check_number(num):
    if num > 0:
        return "Positive"
    elif num == 0:
        return "Zero"
    else:
        return "Negative"

print(check_number(0))   
print(check_number(10))  

# Conditional Statements
x = 10
if x > 0:
    print("x is positive")
else:
    print("x is negative or zero")

# Loop Statements
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

count = 0
while count < 5:
    print(count)
    count += 1

# Control Flow Tools
for i in range(10):
    if i == 5:
        break
    print(i)

for i in range(10):
    if i % 2 == 0:
        continue
    print(i)

for i in range(10):
    if i == 5:
        pass
    print(i)


# nested if-else
x = 10
if x > 0:
    if x % 2 == 0:
        print("x is positive and even")
    else:
        print("x is positive and odd")
        
# if-elif-else
x = 10
if x > 0:
    print("x is positive")
elif x == 0:
    print("x is zero")
else:
    print("x is negative")
    
# Using exec with globals and locals
global_scope = {"x": 5}
local_scope = {}

exec("y = x + 10", global_scope, local_scope)
print(local_scope["y"])  # Output: 15

# Using eval
x = 10
y = eval("x + 10")
print(y)  # Output: 20

