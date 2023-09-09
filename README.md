# SpamFilterNaiveBayas
Questo progetto nasce dall'idea di sfruttare l'intelligenza artificiale, e in particolare il machine learning per poter ovviare al problema delle amil spam che riceviamo tutti i giorni. 
L'idea è quella di utilizzare un classificatore basato su algoritmo Naive Bayes che, analizzando le parole contenute all'interno del testo della mail, riesca a classificare una determinata mail come spam oppure come ham (ovvero non-spam).
Ciò è stato fatto utilizzando come linguaggio di programmazione Pyhton in combinazione con alcuni tool come `scikit-learn` e `pandas`, i quali hanno permesso, rispettivamente, di creare il modello di machine learning e di analizzare il dataset
contenente i dati di 5172 email che sono serviti per addestrare il classificatore. 

Il dataset in questione è stato reperito sulla community Kaggle a questo [link](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv).

Il repository contiene i file .py necessari per poter eseguire il progetto. I file sono `main.py` ovvero il file che dovrà essere eseguito, mentre i file `data_preparation.py` e `agent.py` sono file contenenti classi/funzioni che verranno richiamate
dal main program. 

Affinchè il progetto possa essere eseguito correttamente è necessario installare le seguenti dipendenze:
  - `pip install -U scikit-learn`
  - `pip install pandas`
  - `python -m pip install -U matplotlib`
Una volta installate queste dipendenze basterà eseguire il file `main.py` per poter avviare il main program. 
