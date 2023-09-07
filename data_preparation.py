import pandas as pd
from sklearn.model_selection import train_test_split 



def pull_DataFrame_csv(path) -> pd.DataFrame:
    ds = pd.read_csv(path)
    return ds

def data_cleaning(ds: pd.DataFrame, columns):
    ds = ds.drop(columns=columns)
    return ds

def info(ds: pd.DataFrame):
    ds.info()

def get_column(ds: pd.DataFrame, column: str):
    return ds[column]

""" La funzione train_test_split prene in input due parametri (X, y) che rappresentano rispettivamente la lista delle features (quindi le variabili indipendenti, ovvero
    le colonne) e la lista dei target (ovvero la lista contenente la colonna con il valore che vorremo predire). I restanti parametri sono opzionali
    La funzione restituisce quattro liste, due per ciascuna lista che abbiamo passato come parametro. La prima lista contiene il training set derivato dalle variabili 
    indipendenti, la seconda il test set derivato dalle variabili dipendenti, la terza il training set derivato dalle variabili dipendenti e la quarta il test
    set derivato dalle variabili dipendenti"""
def split_set(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    return train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

def coun_spam_ham(list):
    ham_counter = 0
    spam_counter = 0
    for x in list:
        if x ==0:
            ham_counter = ham_counter + 1
        elif x==1:
            spam_counter = spam_counter + 1

    return ham_counter, spam_counter