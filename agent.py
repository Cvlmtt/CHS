from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
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