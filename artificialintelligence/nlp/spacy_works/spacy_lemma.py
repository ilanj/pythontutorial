import spacy

# Load English tokenizer, tagger,
# parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("""My name is Shaurya Uppal. I enjoy writing
          articles on GeeksforGeeks checkout my other
          article by going to my profile section.""")

doc = nlp(text)

for token in doc:
    print(token, token.lemma_)