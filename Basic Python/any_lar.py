l = input("Enter elements in the list by providing spaces: ").split()
length = len(l) 
cnt = [0] * length # to count the number of elements smaller than each element
for i in range(length):
    c = 0
    for j in range(length):
        if int(l[i]) < int(l[j]):
            c += 1
    cnt[i] = c
for i in range(length):
    if cnt[i] == 1:
        print(int(l[i]), end=" ")
        break
        