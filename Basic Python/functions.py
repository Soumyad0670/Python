class MyClass:
    def __init__(self, value):
        self.value = value 

    def display(self):
        print(f"Value: {self.value}")

obj = MyClass(10)
obj.display()  # Output: Value: 10

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Creating an object of the Person class
person1 = Person("Alice", 30)
person1.greet()  # Output: Hello, my name is Alice and I am 30 years old.

# Method Overloading Example
class MathOperations:
    def add(self, a, b=None, c=None):
        if b is None and c is None:
            return a
        elif c is None:
            return a + b
        else:
            return a + b + c

# Creating an object of MathOperations class
math_ops = MathOperations()

# Calling the add method with different number of arguments
print(math_ops.add(1))        # Output: 1
print(math_ops.add(1, 2))     # Output: 3
print(math_ops.add(1, 2, 3))  # Output: 6

# with arguments and without return keyword
def sum(a,b):
     s=a+b
     print(f"Sum of {a} and {b} is {s}")
sum(9,8)

# without arguments and with return keyword
def sum():
    a=9
    b=8
    s= a+b
    
    return s

sum()

# with arguments and with return keyword

def sum(a,b):
    s= a+b
    
    return s

sum(9,8)

# without arguments and without return keyword

def sum():
    a=9
    b=8
    s= a+b
    print(s)
    
sum()

#Types of functions

# User Defined function
# Built-in function
# Lambda function --> annonymous function

# Built-in function
lst= [2,3,45,2,4,5423,3112,42]
print(len(lst))
print(sum(lst))
print(max(lst))
print(min(lst))

def lenn(data):
    c=0
    for i in data:
        c+=1
        return c
   
print(lenn(lst))
        
x=lambda a,b: a+b
x(17,587)
print(x)

class Animal:
    def sound(self):
        print("This is a generic animal sound.")

class Dog(Animal):
    def sound(self):
        print("Woof! Woof!")

class Cat(Animal):
    def sound(self):
        print("Meow! Meow!")

# Creating objects of the subclasses
dog = Dog()
cat = Cat()

# Calling the overridden methods
dog.sound()  # Output: Woof! Woof!
cat.sound()  # Output: Meow! Meow!

def avg(a,b,c):
    s=(a+b+c)
    average= s/3
    print(average)
    
avg(1,2,3)
