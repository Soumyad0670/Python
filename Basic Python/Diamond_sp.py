def diamond(n):
    a = 1
    print(' ' * (n + 1) + str(a))
    for i in range(1, n):
        a += 10 ** i
        print(' ' * (n - i), end=' ')
        print(a ** 2)
    a = 1
    for i in range(n):
        a += 10 ** (n-i)
    for i in range(n, -1, -1):
        print(' ' * (n - i), end=' ')
        print(a ** 2)
        a = a - 10**i
if __name__ == "__main__":
    n = int(input("Enter a number: "))
    if n < 9 and n >= 1:
        diamond(n)
    else:
        print("Please enter a number between 1 and 8.")
