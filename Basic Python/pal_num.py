def is_palindrome_math(number):
    original = number
    reversed_num = 0
    while number > 0:
        digit = number % 10
        reversed_num = reversed_num * 10 + digit
        number = number // 10
    return original == reversed_num
def check_palindrome():
    try:
        num = int(input("Enter a number to check if it's palindrome: "))
        if num < 0:
            print("Negative numbers are not palindrome")
            return
        is_palindrome_math_result = is_palindrome_math(num)
        if is_palindrome_math_result:
            print(f"{num} is a palindrome number!")
        else:
            print(f"{num} is not a palindrome number!")
    except ValueError:
        print("Please enter a valid number!")
if __name__ == "__main__": 
    check_palindrome()