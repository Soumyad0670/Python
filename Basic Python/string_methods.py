# Splitting a string by whitespace (default behavior)
str1 = "soumyadeep is a good boy"
print(str1.split())  # Output: ['soumyadeep', 'is', 'a', 'good', 'boy']

# Splitting a string by a specific delimiter
str2 = "apple,banana,cherry"
print(str2.split(','))  # Output: ['apple', 'banana', 'cherry']

# Splitting a string with a maximum number of splits
str3 = "one,two,three,four"
print(str3.split(',', 2))  # Output: ['one', 'two', 'three,four']str = "soumyadeep"
print(str3.split())  # Output: ['soumyadeep']

# Additional examples
str1 = "soumyadeep is a good boy"
print(str1.split())  # Output: ['soumyadeep', 'is', 'a', 'good', 'boy']

str2 = "apple,banana,cherry"
print(str2.split(','))  # Output: ['apple', 'banana', 'cherry']

# Joining a list of strings with a comma separator
fruits = ["apple", "banana", "cherry"]
fruit_string = ", ".join(fruits)
print(fruit_string)  # Output: "apple, banana, cherry"

# Joining characters of a string 

words=["Soumyadeep","is","a","good","boy"]
joined_str = " ".join(words)
print(joined_str)  # Output: "soumyadeep is a good boy"

# Additional examples
fruits = ["apple", "banana", "cherry"]
fruit_string = ", ".join(fruits)
print(fruit_string)  # Output: "apple, banana, cherry"

chars = "hello"
hyphenated = "-".join(chars)
print(hyphenated)  # Output: "h-e-l-l-o"