#as set is un ordered u cannot do any operation with index
elements={'a','b','c','d'}
elements.add('e')
elements.add('a')
elements.add(78)
print(elements)

elements.add("new_element")
elements.add(45)
#order changes at every execution
print(elements)
elements.remove('a')
print("after pop operation",elements)

nos={25,89,85}

print(sum(nos))
nos.add(25)
print(nos)
nos.remove(25)
print(nos)
e= next(iter(nos))
print(e)

