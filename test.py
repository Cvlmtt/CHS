import random
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from data_preparation import *
from agent import *

path = "~/bin/SpamDataset2/emails.csv"
columns = ['Email No.']
separetor = "####################################\n"


ds = pull_DataFrame_csv(path)
ds = data_cleaning(ds, columns)

#get dataset info
print(separetor)
print(f"\n\nDataSet Info:\n{separetor}")
info(ds)


#Data separetion in indipendent and dipendent variables
indipendent_variables = data_cleaning(ds, ['Prediction'])
dipendent_variables = get_column(ds, 'Prediction')

#counter of spam and ham emails
ham_counter, spam_counter = coun_spam_ham(dipendent_variables) 

print(separetor)
print(f"All entrys -> Ham emails: {ham_counter}, Spam emails: {spam_counter}\n")

#Data preparation
print(separetor)
print("Splitting DataSet...\n")
indipendent_variables_training_set, indipendent_variables_testing_set, dipendent_variables_training_set, dipendent_variables_test_set = split_set(indipendent_variables, dipendent_variables, test_size=0.33, random_state=random.randint(0, 256))

#counter of spam and ham eamils after split
print(separetor)
ham_counter, spam_counter = coun_spam_ham(dipendent_variables_test_set)
print(f"Test entrys -> Ham emails: {ham_counter}, Spam emails: {spam_counter}\n")

#Traingins
agent = Agent()
print(separetor)
print("Agent training...\n")
agent.fit(indipendent_variables_training_set, dipendent_variables_training_set)

#Testing
print(separetor)
print("Agent predicting...\n")
dipendent_variables_prediction = agent.predict(indipendent_variables_testing_set) 


#Evalutating
print(separetor)
print("Agent evaluating...")

lables = [0, 1]
accurancy, precision, recall, cm , mcc= agent.valuation(dipendent_variables_test_set, dipendent_variables_prediction, labels=lables)

print(f"Accurancy:{'%.2f'%(accurancy*100)}%\nPrecision:{'%.2f'%(precision*100)}%\nRecall:{'%.2f'%(recall*100)}%\nMCC: {'%.2f'%(mcc*100)}%")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lables)
disp.plot()

cv_score = agent.cross_validation(X_train=indipendent_variables_training_set, y_train=dipendent_variables_training_set)
print(cv_score)
plt.show()