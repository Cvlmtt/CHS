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
from data_preparation import *
from agent import *

path = "~/bin/SpamDataset2/emails.csv"
ds = pull_DataFrame_csv(path)
columns = ['Email No.']
separetor = "####################################\n"
ds = data_cleaning(ds, columns)

print(f"\n\nDataSet Info:\n{separetor}")
info(ds)
print(separetor)

indipendent_variables = data_cleaning(ds, ['Prediction'])
dipendent_variables = get_column(ds, 'Prediction')


ham_counter, spam_counter = coun_spam_ham(dipendent_variables) 
print(separetor)
print(f"All entrys -> Ham emails: {ham_counter}, Spam emails: {spam_counter}\n")
print(separetor)

print("Splitting DataSet...\n")
indipendent_variables_training_set, indipendent_variables_testing_set, dipendent_variables_training_set, dipendent_variables_test_set = split_set(indipendent_variables, dipendent_variables, test_size=0.33, random_state=random.randint(0, 256))

print(separetor)
ham_counter, spam_counter = coun_spam_ham(dipendent_variables_test_set)
print(f"Test entrys -> Ham emails: {ham_counter}, Spam emails: {spam_counter}\n")
print(separetor)

agent = Agent()
print("Agent training...")
agent.fit(indipendent_variables_training_set, dipendent_variables_training_set)

print(separetor)
print("Agent predicting...")
dipendent_variables_prediction = agent.predict(indipendent_variables_testing_set) 
print(separetor)

print("Agent evaluating...")

accurancy = accuracy_score(dipendent_variables_test_set, dipendent_variables_prediction)
print(f"Accurancy: {accurancy}")
precision = precision_score(dipendent_variables_test_set, dipendent_variables_prediction)
print(f"Precision: {precision}")
recall = recall_score(dipendent_variables_test_set, dipendent_variables_prediction)
print(f"Recall: {recall}")
print(separetor)

lables = [0, 1]
cm = confusion_matrix(dipendent_variables_test_set, dipendent_variables_prediction, labels=lables)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lables)
disp.plot()
plt.show()