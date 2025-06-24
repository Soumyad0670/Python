file=open('input.py', mode='r')
print(file.read())
print(file.close())

with open('input.py', mode='r') as file:
    print(file.read())  
    print(file.close())
    
with open('D:\Python', mode='r') as file:
    print(file.read())
    
with open('a.txt', mode='w') as file:
    file.write('Hello World!')
    print(file.write())
    
with open('a.txt', mode='a') as file:
    file.append('new txt file!')
    print(file.readlines())
    
with open('a.txt', mode='r') as file:
    print(file.read())

with open('a.txt', mode='w') as file:
    print(file.write())
    
import os
print(os.path.exists('a.txt'))