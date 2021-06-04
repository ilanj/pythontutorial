import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

corpus = 'Let us understand the difference between sentence & word tokenizer. It is going to be a simple example. '

#word tokenizer
tokens = word_tokenize(corpus)
print(type(tokens), len(tokens), tokens)

#sentence tokenizer
sentences = sent_tokenize(corpus)
print(sentences)

#stopwords
english_stops = stopwords.words('english')
#extending stopwords
english_stops.extend(['from', 'subject', 're', 'edu', 'use'])

#wordnet
from nltk.corpus import wordnet as wn

syn = wn.synsets('dog')[0]
print(syn)
print(syn.hypernyms()[0].hyponyms())
pass
