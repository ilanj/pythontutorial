def isPowerOfThree(n):
    #return false for less than 3
    if n < 3:
        return False
    flag = False
    while(n):
        if n % 3 == 0:
            q = n % 3
            n = n / 3
            #only for power of 3's n will become 1
            if n == 1:
                return True
        else:
            return False

#testing
for i in range(500):
    if isPowerOfThree(i):
        print(i, isPowerOfThree(i))