def armstrong(num): # Function to check if a number is armstrong
    original = num
    s = 0
    while num > 0:
        digit = num % 10 
        s = s + (digit ** 3)
        num = num // 10
    return s == original
if __name__ == '__main__': # This is the main function
        number = int(input("Enter a number to check if it's armstrong: "))
        if number < 0:
            print("Negative numbers are not armstrong")
        if armstrong(number):
            print(f"{number} is a armstrong number!")
        else:
            print(f"{number} is not a armstrong number!")
