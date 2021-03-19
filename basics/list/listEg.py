import random

nos= []
rev= list()
for i in range(1,5):
    # nos.append(eval(input("enter any input")))
    # nos.append(i)
    nos.append(random.randint(10,99))
print("sum=",sum(nos))
print("sum=",sum(nos,20))
l=len(nos)

print(nos)
print(nos[3:4])
print(l)
for i in range(l):
    print(nos[i])
print(nos[1:3])
print(nos)
nos.append(36)
print(nos)
addsum=sum(nos, 10)
print(addsum)
print(sum(nos))
#print(nos.sort())
rev=list(reversed(nos))

for a in reversed(rev):
    print('after reverse',a)
for a in nos:
    print('without reverse',a)

sorted=list(sorted(nos))
print("after sort=",sorted)
nos.append(10)
nos.append(10)
nos.append(10)
print(nos)