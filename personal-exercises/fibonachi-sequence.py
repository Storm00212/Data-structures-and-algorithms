prev1 = 0 
prev2 = 1
n = 10
print("fibonacchi sequence in python")
for i in range(n):
    current = prev1 + prev2
    prev1 = prev2
    prev2 = current
    
    for j in range(current):
        print('*', end='')
    print()  # Add a line break after printing the asterisks


