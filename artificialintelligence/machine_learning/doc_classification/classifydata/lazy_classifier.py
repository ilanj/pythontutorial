import numpy as np
import lazypredict
import joblib
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_files, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


DATA_DIR = "/home/ila/Documents/repos/python-works/artificialintelligence/machine_learning/doc_classification/classifydata/dataset_5classes/"
# DATA_DIR = "/home/ila/Documents/900_docs/ocr_text/"

# data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace")
data = load_breast_cancer()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
models