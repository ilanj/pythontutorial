import nltk
with open("sample.txt") as file:
    data_text = file.read()

sentences = nltk.sent_tokenize(data_text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
for sent in tagged_sentences:
    print( nltk.ne_chunk(sent))