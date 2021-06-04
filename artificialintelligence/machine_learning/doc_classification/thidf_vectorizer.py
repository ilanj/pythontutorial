from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["I like apples from trees to eat and to make juice to family", "I Like OraNges for juice"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names(), "\n\n")
for index in range(X.data.size - 1):
    print("word = ", vectorizer.get_feature_names()[index],"\t idf = ", vectorizer.idf_[index], "\t tfidf-idf = ",X.data[index])
print("\n\n", X.toarray())