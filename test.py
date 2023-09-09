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


#GaussianNB
#Traingins
print("Gaussian Naive Bayes agent:")
print(separetor)
gaussian_agent = Agent(model_type="Gaussian")
print(separetor)
print("Agent training...\n")
gaussian_agent.fit(indipendent_variables_training_set, dipendent_variables_training_set)

#Testing
print(separetor)
print("Agent predicting...\n")
dipendent_variables_prediction = gaussian_agent.predict(indipendent_variables_testing_set) 


#Evalutating
print(separetor)
print("Agent evaluating...")

lables = [0, 1]
g_accurancy, g_precision, g_recall, g_cm , g_mcc = gaussian_agent.valuation(dipendent_variables_test_set, dipendent_variables_prediction, labels=lables)

print(f"Gaussian Accurancy:{'%.2f'%(g_accurancy*100)}%\nGaussian Precision:{'%.2f'%(g_precision*100)}%\nGaussian Recall:{'%.2f'%(g_recall*100)}%\nGaussian MCC: {'%.2f'%(g_mcc*100)}%")
print(separetor)

disp = ConfusionMatrixDisplay(confusion_matrix=g_cm, display_labels=lables)
disp.plot()


g_fit_time_mean, g_score_time_mean, g_accuracy_mean, g_precision_mean, g_recall_mean = gaussian_agent.cross_validation(X_train=indipendent_variables_training_set, y_train=dipendent_variables_training_set)
print(separetor)
print("Cross validatio means:\n")
print(f"Gaussian Fit time mean:{g_fit_time_mean}\nGaussian Score time mean:{g_score_time_mean}\nGaussian Accuracy test mean: {g_accuracy_mean}\nGaussian Precision test mean: {g_precision_mean}\nGaussian Recall test mean:{g_recall_mean}")
print(separetor)
print(g_cm)
plt.show()


#BernoulliNB
#Traingins
print("Bernoulli Naive Bayes agent:")
bernoulli_agent = Agent(model_type="Bernoulli")
print(separetor)
print("Agent training...\n")
bernoulli_agent.fit(indipendent_variables_training_set, dipendent_variables_training_set)

#Testing
print(separetor)
print("Agent predicting...\n")
dipendent_variables_prediction = bernoulli_agent.predict(indipendent_variables_testing_set) 


#Evalutating
print(separetor)
print("Agent evaluating...")

lables = [0, 1]
b_accurancy, b_precision, b_recall, b_cm , b_mcc = bernoulli_agent.valuation(dipendent_variables_test_set, dipendent_variables_prediction, labels=lables)

print(f"Bernoulli Accurancy:{'%.2f'%(b_accurancy*100)}%\nBernoulli Precision:{'%.2f'%(b_precision*100)}%\nBernoulli Recall:{'%.2f'%(b_recall*100)}%\nBernoulli MCC: {'%.2f'%(b_mcc*100)}%")
print(separetor)

disp = ConfusionMatrixDisplay(confusion_matrix=b_cm, display_labels=lables)
disp.plot()


b_fit_time_mean, b_score_time_mean, b_accuracy_mean, b_precision_mean, b_recall_mean = bernoulli_agent.cross_validation(X_train=indipendent_variables_training_set, y_train=dipendent_variables_training_set)
print(separetor)
print("Cross validatio means:\n")
print(f"Bernoulli Fit time mean:{b_fit_time_mean}\nBernoulli Score time mean:{b_score_time_mean}\nBernoulli Accuracy test mean: {b_accuracy_mean}\nBernoulli Precision test mean: {b_precision_mean}\nBernoulli Recall test mean:{b_recall_mean}")
print(separetor)
print(b_cm)
plt.show()

#ComplementNB
#Traingins
print("Complement Naive Bayes agent:")
complement_agent = Agent(model_type="Complement")
print(separetor)
print("Agent training...\n")
complement_agent.fit(indipendent_variables_training_set, dipendent_variables_training_set)

#Testing
print(separetor)
print("Agent predicting...\n")
dipendent_variables_prediction = complement_agent.predict(indipendent_variables_testing_set) 


#Evalutating
print(separetor)
print("Agent evaluating...")

lables = [0, 1]
c_accurancy, c_precision, c_recall, c_cm , c_mcc= complement_agent.valuation(dipendent_variables_test_set, dipendent_variables_prediction, labels=lables)

print(f"Complement Accurancy:{'%.2f'%(c_accurancy*100)}%\nComplement Precision:{'%.2f'%(c_precision*100)}%\nComplement Recall:{'%.2f'%(c_recall*100)}%\nComplement MCC: {'%.2f'%(c_mcc*100)}%")
print(separetor)

disp = ConfusionMatrixDisplay(confusion_matrix=c_cm, display_labels=lables)
disp.plot()


c_fit_time_mean, c_score_time_mean, c_accuracy_mean, c_precision_mean, c_recall_mean = complement_agent.cross_validation(X_train=indipendent_variables_training_set, y_train=dipendent_variables_training_set)
print(separetor)
print("Cross validatio means:\n")
print(f"Complement Fit time mean:{c_fit_time_mean}\nComplement Score time mean:{c_score_time_mean}\nComplement Accuracy test mean: {c_accuracy_mean}\nComplement Precision test mean: {c_precision_mean}\nComplement Recall test mean:{c_recall_mean}")
print(separetor)
print(c_cm)
plt.show()