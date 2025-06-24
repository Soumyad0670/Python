def print_pascal_triangle(n):
    triangle = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            if j == 0 or j == i:
                row.append(1)
            else:
                row.append(triangle[i-1][j-1] + triangle[i-1][j])
        triangle.append(row)
    for i in range(n):
        print(" " * (n - i - 1), end="")
        for j in triangle[i]:
            print(j, end=" ")
        print()
def main():
    n = int(input("Enter the number of rows for Pascal's Triangle: "))
    print_pascal_triangle(n)
if __name__ == "__main__":
    main()





