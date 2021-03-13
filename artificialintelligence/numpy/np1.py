import numpy as np

x = np.array([1, 2, 3, 4, 5])

print("x=", x)

print('type of x=', type(x))

print('datatype of x=', x.dtype)

print('shape of x=', x.shape)

y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print('y=', y)
print('shape of y=', y.shape)

print('size of y=', y.size)
print('datatype of y=', y.dtype)
z_str = np.array(['hi', 'welcome'])
print(z_str)

print("str.dtype ", z_str.dtype)

nos = np.array([1, 2, 'hi'])
print("nos.dtype ",nos.dtype)

mixed_nos = np.array([1, 2, 3.4534, 3.3333])
print('mixednos.dtype ',mixed_nos.dtype)
print('mixed_nos=',mixed_nos)

mixed_nos1 = np.array([1, 2, 3.4534, 3.3333], dtype=np.int64)
print('mixednos1.dtype ',mixed_nos1.dtype)
print("mixed_nos1= ", mixed_nos1)


temp = np.array([9, 8, 7, 6])
np.save('newarray', temp)

myarray = np.load('newarray.npy')
print(myarray)
