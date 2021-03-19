'''
using pool we create multi processes by ppssing as argument. if we are going to pass 5 numbers in list, pool size has to be >= 5.
if the pool size is 5, and list size is 6 to 10 , it will be called twice, if greater than 10 than will be called thrice
'''
import time
from multiprocessing import Pool

def f(x):
    time.sleep(2)
    # if x == 7:
    #     time.sleep(6)
    return x*x

if __name__ == '__main__':
    nos = [1, 2, 3, 4, 4 ,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                           4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                           4,4,4,4,4,4,4,4,4,4,4 ]
    print("list size = ",len(nos))
    with Pool(60) as p:
        result = p.map(f, nos)
        print(result)