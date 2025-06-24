# Taking input from the user
name = input("Enter your name: ")
print(f"Hello, {name}!")

# Taking numerical input from the user

num = int(input("Enter a number: "))
print(f"The number you entered is {num} and its square is {num ** 2}")

# Taking float input from the user
num = float(input("Enter a decimal number: "))
print(f"The number you entered is {num} and its square is {num ** 2}")


# Using eval to evaluate a list expression# Use ast.literal_eval for safer evaluation of input
import ast

# Get user input and safely convert it to a list
list_input = input("Enter a list (e.g., [1, 2, 3]): ")

try:
    # Use ast.literal_eval instead of eval for security
    evaluated_list = ast.literal_eval(list_input)

    if isinstance(evaluated_list, list):
        # Cube each element in the list
        cubed_list = [x**3 for x in evaluated_list]
        print("Cubed list:", cubed_list)
    else:
        print("Please enter a valid list.")

except (ValueError, SyntaxError):
    print("Invalid input. Please enter a list like [1, 2, 3].")


