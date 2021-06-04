'''
req:{
pip install spacy
python -m spacy download en_core_web_sm
'''
import spacy
nlp = spacy.load('en_core_web_sm')

sentence = "John Wick from Apple is looking at buying U.K. startup for $1 billion. St Johns Street is were John Wick is residing"
doc = nlp(sentence)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)