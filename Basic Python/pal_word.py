def palindrome_str(str,n):
    return str == str[::-1]
def check_pal():
    word=str(input("Enter a string: "))
    pal=palindrome_str(word)
    if pal:
        print(f"The palindrome word is: {word}")
    else:
        print("The word is not palindrome")
if __name__ == "__main__": 
    check_pal()