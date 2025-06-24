# While loop example
count = 0
while count < 5:
    print(count)
    count += 1

# Break statement example
# break is used to exit a for loop or a while loop
count = 0
while count < 10:
    print(count)
    if count == 5:
        break
    count += 1

# Continue statement example
# continue is used to skip the current block and return to the "for" or "while" statement
count = 0
while count < 10:
    count += 1
    if count % 2 == 0:
        continue
    print(count)
    
l= [1," harry", 2, "ron", 3, "hermione"]
i=0
while i<len(l):
    print(l[i])
    i=i+1
else:
    print("done")
    
    m=0;
    while m<10:
        print(m)
        m+=1
        
# pass is a null statement in Python. Nothing happens when it is executed. It is used as a placeholder.
for i in range(10):
    if i == 5:
        pass
    print(i)