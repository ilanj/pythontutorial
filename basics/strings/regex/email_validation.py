'''
email should start with no and end with alphabet
34subhahsih@gmail.com is correct
56ram4566@gmail.com is wrong
'''

import re

text = "34subhahsih@gmail.com is correct \n 56ram4566@gmail.com is wrong"
# phone_no_regex = re.compile(r"\d\d\d-\d\d\d-\d\d\d\d")
email_valid = re.compile(r".*[a-z]\@gmail\.com")
match = email_valid.findall(text)
print(match)