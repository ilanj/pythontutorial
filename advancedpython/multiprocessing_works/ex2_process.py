from multiprocessing import Process

def printname(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=printname, args=('bob',))
    p.start()
    p.join()