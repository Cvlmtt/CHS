import random
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt


ds = pd.read_csv("~/bin/SpamDataset2/emails.csv")

ds = ds.drop(columns=['Email No.'])

ds.info()

indipendent_variables = ds.drop('Prediction', axis=1)
dipendent_variable = ds['Prediction'] 

spam_counter = 0
ham_counter = 0
for x in dipendent_variable:
    if x == 0:
        ham_counter = ham_counter + 1
    elif x == 1:
        spam_counter = spam_counter + 1 

print(f"All entrys -> Ham emails: {ham_counter}, Spam emails: {spam_counter}")

X_train, X_test, y_train, y_test = train_test_split(indipendent_variables, dipendent_variable, test_size=0.33, random_state=random.randint(0, 256))

spam_counter = 0
ham_counter = 0
for x in y_test:
    if x == 0:
        ham_counter = ham_counter + 1
    elif x == 1:
        spam_counter = spam_counter + 1 

print(f"Test entrys -> Ham emails: {ham_counter}, Spam emails: {spam_counter}")

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accurancy = accuracy_score(y_test, y_pred)
print(f"Accurancy: {accurancy}")
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision}")
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")

lables = [0, 1]
cm = confusion_matrix(y_test, y_pred, labels=lables)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lables)
disp.plot()
plt.show()