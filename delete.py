a = [2,3,6,7,2,9,0]
b =a [:]
[print(x,y) if x+y == 9 for x,y in list(zip(a,b))]