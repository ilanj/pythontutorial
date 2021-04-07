import re

txt = "I like 5^*&^*%&%&^%&^%^%$%5% bananas"
print(re.sub(r"[^a-zA-Z\s]", "", txt))

# txt = re.sub(r"[^a-zA-Z]", "", txt)
#
# print(txt)
# text = "$25,400.00"
# print(re.sub(r"\D", "", text)[:-2])

text = "dd"
print(text.join("df"))