import pickle
import time

import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_files
from sklearn.externals import joblib
from sklearn.feature_extraction import stop_words
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from artificialintelligence.machine_learning.doc_classification.classifydata.Helper import Test

DATA_DIR = "dataset_5classes"

data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace")
#print("the classes are",data.target_names)
metadata = np.unique(data.target, return_counts=True)

labels=metadata[0]
count=metadata[1]
names=data.target_names
#print(dict(zip(names,count)))
for i in range(len(labels)):
    print(labels[i],names[i],count[i])

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.10)#test_size=0.50
print("X_test.shape=",len(X_test))
print("Y_test.shape=",y_test.shape)
print("X_train.shape=",len(X_train))
print("y_train.shape=",y_train.shape)
check=dict(zip(X_train,y_train))


vectorizer = TfidfVectorizer(stop_words="english", max_features=2000, decode_error="ignore")
# print("Stop Words=",vectorizer.stop_words_)
# with open("stopwords.txt") as fptr:
#     fptr.write(vectorizer.stop_words_)
vectorizer.fit(X_train)
keywords=vectorizer.get_feature_names()
print("keywords=",keywords)
#print("keywords=",keywords)
#print("keywords size=",len(keywords))
X_train_vectorized = vectorizer.transform(X_train)
# print("Stop Words=",vectorizer.stop_words_)
with open("stopwords.txt", "w") as fptr:
    for word in vectorizer.stop_words_:
        fptr.write(word+" ")
    for word in stop_words.ENGLISH_STOP_WORDS:
        fptr.write(word + " ")
with open("features.txt", "w") as fptr:
    for word in keywords:
        fptr.write(word + " ")

#naivebayes
cls = MultinomialNB(alpha=0.001)
cls = clf = RandomForestClassifier()
cls.fit(vectorizer.transform(X_train), y_train)
joblib.dump(cls, "" + 'nb_model.h5')
testobj=Test()
test_doc=testobj.data_for_a_document()
y_pred = cls.predict(vectorizer.transform(X_test))
print("accuracy with naive bayes = ",accuracy_score(y_test, y_pred))
#RandomForest
clf = clf = RandomForestClassifier()
clf.fit(vectorizer.transform(X_train), y_train)
joblib.dump(clf, "" + 'rf_model.h5')
y_pred = clf.predict(vectorizer.transform(X_test))
print("accuracy with randomforest classifier = ",accuracy_score(y_test, y_pred))

svc_tfidf = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=3000)),
    ("linear svc", SVC(kernel="linear"))])

models = [("svc_tfidf", svc_tfidf), ]

svc_score = [(name, cross_val_score(model, X_train, y_train, cv=2).mean()) for name, model in models]
scores = sorted(svc_score, key=lambda x: -x[1])
print(scores)
