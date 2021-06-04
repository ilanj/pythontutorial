import nltk
from nltk.stem import PorterStemmer
word_stemmer = PorterStemmer()
root_word = word_stemmer.stem('was')
print(root_word)
