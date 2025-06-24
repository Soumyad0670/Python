age = 21
if (age >= 18):
    print("I am eligible")
else:
    print("Not eligible")

def check_number(num):
    if num > 0:
        return "Positive"
    elif num == 0:
        return "Zero"
    else:
        return "Negative"

print(check_number(0))   
print(check_number(-5))  
print(check_number(10))

# Using enumerate with a list
# The enumerate function in Python adds a counter to an iterable and returns it as an enumerate object.
# This is useful when you need both the index and the value of each item in an iterable during a loop.
# The enumerate function takes two arguments:
# iterable: The iterable you want to loop over.
# start: The index value from which the counter should start. By default, it is 0.
# The enumerate function returns an enumerate object that contains the index and value of each item in the iterable.
# You can convert the enumerate object to a list, tuple, or dictionary, or use it directly in a loop.

fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate (fruits):
    print(index, fruit)
    
s="Soumyadeep"
for index, char in enumerate(s):
    # print(index ,char)
    print(str(index)+ "-->"+ char )
print(s.__len__())





