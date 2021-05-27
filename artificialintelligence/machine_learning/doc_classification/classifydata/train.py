import pickle
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_files
import joblib
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from artificialintelligence.machine_learning.doc_classification.classifydata.Helper import Test
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import plot_confusion_matrix

DATA_DIR = "dataset_5classes"

data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace")
#print("the classes are",data.target_names)
metadata = np.unique(data.target, return_counts=True)

labels=metadata[0]
count=metadata[1]
class_names=data.target_names
#print(dict(zip(names,count)))
for i in range(len(labels)):
    print(labels[i], class_names[i], count[i])

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, random_state = False)#test_size=0.50
print("X_test.shape=",len(X_test))
print("Y_test.shape=",y_test.shape)
print("X_train.shape=",len(X_train))
print("y_train.shape=",y_train.shape)
check=dict(zip(X_train,y_train))

#count vectorizer for BOW, just implemented and not used
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
#tfidf -used here
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
print("Stop Words=",vectorizer.stop_words_)
with open("stopwords.txt", "w") as fptr:
    for word in vectorizer.stop_words_:
        fptr.write(word+" ")
    for word in ENGLISH_STOP_WORDS:
        fptr.write(word + " ")
with open("features.txt", "w") as fptr:
    for word in keywords:
        fptr.write(word + " ")

#naivebayes
cls = MultinomialNB(alpha=0.001)
clf = RandomForestClassifier(n_estimators = 5) #ensemble method
cls.fit(vectorizer.transform(X_train), y_train)
joblib.dump(cls, "" + 'nb_model.h5')
testobj=Test()
test_doc=testobj.data_for_a_document()
y_pred = cls.predict(vectorizer.transform(X_test))
print("accuracy with naive bayes = ",accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#RandomForest
clf = RandomForestClassifier()
clf.fit(vectorizer.transform(X_train), y_train)
joblib.dump(clf, "" + 'rf_model.h5')
y_pred = clf.predict(vectorizer.transform(X_test))
print("accuracy with randomforest classifier = ",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

svc_tfidf = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=3000)),
    ("linear svc", SVC(kernel="linear"))])

models = [("svc_tfidf", svc_tfidf), ]

svc_score = [(name, cross_val_score(model, X_train, y_train, cv=2).mean()) for name, model in models]
scores = sorted(svc_score, key=lambda x: -x[1])
print(scores)

models = []
models.append(('LogisticRegression', LogisticRegression(solver = 'lbfgs', max_iter=120)))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
# models.append(('GaussianNB', GaussianNB()))
# models.append(('SVC', SVC()))


# Loop through models and identify the best model to predict square
for name, model in models:
    model.fit(vectorizer.transform(X_train), y_train)
    # x_predict = 7
    y_pred = clf.predict(vectorizer.transform(X_test))
    print(name, accuracy_score(y_test, y_pred))

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(cls, vectorizer.transform(X_test), y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
plt.savefig("cf.png", format='png')