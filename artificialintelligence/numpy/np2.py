import numpy as np
nos=[1,2,3,4,5,6]
print(type(nos))
print(type(np.array(nos)))
print(np.array(nos))

x=np.ones((3,3))  #all r and c  as 1
print(x)
# x=np.eye(3)
x=np.eye(4)  #diagonals will be 1 and rest as 0
print(x)

dgnl=np.diag([2,4,6,5,4,7])   #these nos will be in diaonal and rest will be 0's
print(dgnl)
print("np.arrange(10)",np.arange(10))
print("np.arange(10,13)",np.arange(10,13))
print("np.arange(10,100,5)",np.arange(10,100,5))
print(np.linspace(1,10,21))#takes 20 nos between 1 to 10

n=np.arange(12)
n1=np.reshape(n,(3,4))#exactly reshape with 12 elements
print(n1)
# The endpoint of the interval can optionally be excluded.
n1=np.linspace(0,10,25,endpoint=False).reshape(5,5)
print("n1= ",n1)

n1=np.random.randint(10,99,(3,3))
print(n1)

n1=np.random.normal(0,0.1,size=(250,250))#creates a 250x250 array with mean 0 and sd 0.1
print(n1)
print('mean',n1.mean())
print('std dev',n1.std())
print('max',n1.max())
print('min',n1.min())
print('sum > 0',(n1>0).sum())
print('sum < 0',(n1<0).sum())
matrix=np.linspace(2,32,16).reshape(4,4)
print(np.linspace(2,32,16).reshape(4,4))


