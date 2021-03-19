char_set=set()

n=int(input("enter how many chars u want to enter"))

for i in range(n):
    char_set.add(eval(input("enter a no")))
print(char_set)
print(sum(char_set))
print(frozenset(sorted(char_set)))
print(sorted(sorted(char_set)))
# print(reversed((char_set)))#will not work


