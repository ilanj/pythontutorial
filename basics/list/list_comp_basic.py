nos = [1,2,3,4,5,6,7,8]

nos1 = [(n*n) for n in nos if n > 3]
pass

nos2 = list()
for n in nos:
    if n > 3:
        nos2.append(n*n)

print(nos2)