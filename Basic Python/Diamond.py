def number_pattern(n):
    print("\nAsterisk Diamond Pattern:\n")
    for i in range(n):
        print(' ' * (n - i - 1) + '*' * (2 * i + 1))
    for i in range(n - 1, -1, -1):
        print(' ' * (n - i - 1) + '*' * (2 * i + 1))
    print("\nNumber Diamond Pattern:\n")
    for i in range(1, n + 1):
        print(' ' * (n - i), end='')
        for j in range(1, i + 1):
            print(j, end='')
        for j in range(i - 1, 0, -1):
            print(j, end='')
        print()  
    for i in range(n - 1, 0, -1):
        print(' ' * (n - i), end='')
        for j in range(1, i + 1):
            print(j, end='')
        for j in range(i - 1, 0, -1):
            print(j, end='')
        print()
if __name__ == "__main__":
    n = int(input("Enter a number: "))
    number_pattern(n)