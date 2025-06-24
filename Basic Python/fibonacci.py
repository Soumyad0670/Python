def fibonacci():
    a, b = 0, 1 # Initialize the first two Fibonacci numbers
    while n > 0: # Generate Fibonacci numbers 
        yield a # Yield the current Fibonacci number
        a, b = b, a + b # Update to the next Fibonacci number
if __name__ == "__main__": # Main function to run the generator
    fib_gen = fibonacci() # Create a generator object
    n=int(input("Enter the number of Fibonacci numbers to generate: "))
    for i in range(n):
        print(next(fib_gen)) # Print the next Fibonacci number

# By recursion
def fibonacci(n):
    if n == 1:
        return [0]
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    return fib_sequence
def main():
        n = int(input("Enter the number of Fibonacci numbers to generate: "))
        result = fibonacci(n)
        print("Fibonacci Sequence:")
        for i, num in enumerate(result):
            print(num)
def main():
    main()