from nltk import word_tokenize
from nltk.corpus import stopwords

text = " Hi, I am from Pondicherry and taking python-class from 7:00 to 7.45"
text = text.lower()
text_list = word_tokenize(text)
print(text_list)

#removing stopwords
stop_words = stopwords.words('english')
text_wo_stopwords = list()
for word in text_list:
    if word not in stop_words:
        text_wo_stopwords.append(word)

text = text.replace(r"\w", "")
print(text_wo_stopwords)
pass