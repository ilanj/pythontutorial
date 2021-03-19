sq = list()
for x in range(10):
    sq.append(x**2)
print(sq)
#[expression for (zero or more for or if)]
even = [(x) for x in sq if x%2==0 ]
print(even)



#the above can be expanded as
odd=[]
for x in sq:
    if x%2==1:
        odd.append(x)
print(odd)

#4th op
x2=[x*2 for x in sq]
print(x2)

#5th op
cubes=[(x,x**3) for x in range(6)]
print(cubes)



