from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I like apples", "I Like OraNges"]

print("-----------------uni gram ----------------")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print("get_feature_names = ",vectorizer.get_feature_names())
print(X.toarray())

#bi-gram
print("-----------------bi gram ----------------")
vectorizer2 =CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
print("get_feature_names = ",vectorizer2.get_feature_names())
print(X2.toarray())
