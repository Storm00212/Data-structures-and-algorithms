prev1 = 0 
prev2 = 1
n = 30

for i in range(n):
    print("fibonacchi sequence in python")
    for j in range(current):
        print('*', end='')
    
    current = prev1 + prev2
    prev1 = prev2
    prev2 = current
    