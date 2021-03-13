#whether a no is divisible by both/or 3 nd 5
n = int(input('enter n'))

if n%3==0 and n%5==0:
    print(n,' is divisible by both 3 and 5')
elif n%3==0 or n%5==0:
    print(n,' is divisible by either 3 or 5')
else:
    print(n,' not divisible by both 3 and 5')

