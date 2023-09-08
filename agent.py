import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    
)

class Agent:
    model = None


    def __init__(self) -> None:
        self.model = GaussianNB()


    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def valuation(self, y_test, pred, labels=None):
        accurancy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=labels)
        mcc = matthews_corrcoef(y_test, pred)
        return accurancy, precision, recall, cm, mcc
    
    def cross_validation(self, X_train, y_train):
        rfk = RepeatedKFold(n_splits=10, n_repeats=4, random_state=random.randint(0,256))
        tests=list(["accuracy", "precision", "recall"])
        cv_scores = cross_validate(self.model, X_train, y_train, cv=rfk, n_jobs=3, verbose=5, scoring=tests)
        return cv_scores