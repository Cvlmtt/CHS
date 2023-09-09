import random
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    
)
import statistics


class Agent:
    model = None


    def __init__(self, model_type="Complement") -> None:
        if(model_type == "Complement"):
            print("Complement NB implemented")
            self.model = ComplementNB()
        elif(model_type == "Gaussian"):
            print("Gaussian NB implemented")
            self.model = GaussianNB()
        elif(model_type == "Bernoulli"):
            print("Bernoulli NB implemented")
            self.model = BernoulliNB()


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
        cv_score = cross_validate(self.model, X_train, y_train, cv=rfk, n_jobs=3, verbose=5, scoring=tests)
        fit_time_mean = statistics.mean(cv_score['fit_time'])
        score_time_mean = statistics.mean(cv_score['score_time'])
        accuracy_mean = statistics.mean(cv_score['test_accuracy'])
        precision_mean = statistics.mean(cv_score['test_precision'])
        recall_mean = statistics.mean(cv_score['test_recall'])
        return fit_time_mean, score_time_mean, accuracy_mean, precision_mean, recall_mean