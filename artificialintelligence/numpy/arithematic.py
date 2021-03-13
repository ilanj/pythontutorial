import numpy as np

x=np.random.randint(1,10,size=(3,3))
y=np.random.randint(1,10,size=(3,3))
print(x)
print(y)

# print(x+y)
# print(np.add(x,y))

print(x*y)
print(x/y)

print("sqrt",np.sqrt(x))
print("exponent",np.exp(x))
print("pow 2",np.power(x,2))
print(x.mean())
print(x.mean(axis=0))
print(x.mean(axis=1))
print("std dev",x.std())
print("mean",np.median(x))

print("max ",x.max())
print("min ",x.min())
print(100-x)#100*x, 100/x

x1=np.arange(0,3)
print(x)
print(x1)
print(x+x1)


