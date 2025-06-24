def print_pyramid(n):
    for i in range(n):
        print(' ' * (n-i-1) + '*' * (2*i+1))
def print_hollow_pyramid(n):
    for i in range(n):
        if i == n-1:
            print('*' * (2*n-1))
        else:
            print(' ' * (n-i-1) + '*' + ' ' * (2*i-1) + ('*' if i > 0 else ''))
def print_diamond(n):
    for i in range(n):
        print(' ' * (n-i-1) + '*' * (2*i+1))
    for i in range(n-2, -1, -1):
        print(' ' * (n-i-1) + '*' * (2*i+1))
def print_hollow_diamond(n):
    for i in range(n):
        print(' ' * (n-i-1) + '*' + ' ' * (2*i-1) + ('*' if i > 0 else ''))
    for i in range(n-2, -1, -1):
        print(' ' * (n-i-1) + '*' + ' ' * (2*i-1) + ('*' if i > 0 else ''))
def print_star_square(n):
    for i in range(n):
        print('*' * n)
def print_hollow_square(n):
    for i in range(n):
        if i == 0 or i == n-1:
            print('*' * n)
        else:
            print('*' + ' ' * (n-2) + '*')
def main():
    size = 5  
    print("\nPyramid Pattern:")
    print_pyramid(size)    
    print("\nHollow Pyramid Pattern:")
    print_hollow_pyramid(size)
    print("\nDiamond Pattern:")
    print_diamond(size)    
    print("\nHollow Diamond Pattern:")
    print_hollow_diamond(size)
    print("\nSquare Pattern:")
    print_star_square(size)    
    print("\nHollow Square Pattern:")
    print_hollow_square(size)
if __name__ == "__main__":
    main()