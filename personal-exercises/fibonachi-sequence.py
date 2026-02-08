prev1 = 0 
prev2 = 1
n = 100

for i in range(n):
    current = prev1 + prev2
    prev1 = prev2
    prev2 = current
    print('*' * current)