# try
# except
# finally
try:
    n=int(input("Enter a number: "))    
    result=100/n
    print(result)
except Exception as e:
    print(e)   
finally:
    print("End of program")
    
