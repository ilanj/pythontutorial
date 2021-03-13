import multiprocessing as mp
import time


def foo(q,n):
    time.sleep(3)
    q.put(n*n)

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p1 = ctx.Process(target=foo, args=(q,4,))
    p2 = ctx.Process(target=foo, args=(q,5,))
    p1.start()
    p2.start()
    print(q.get())
    print(q.get())
    # p.join()