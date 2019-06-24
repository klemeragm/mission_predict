# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
#import numpy as pd
base = pd.read_csv("Mission_Prediction_Dataset.csv")
base.describe()
base.head()

#defining the previsors, in portuguese previsores, and de test set with classe
previsores= base.iloc[:, 1:13].values
classe = base.iloc[:, 14].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axix=0)
imputer = imputer.fit(previsores[:, 1:13])
previsores[:, 1:13] =  imputer.transform(previsores[:, 1:13])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores= scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_training, previsores_test, classe_training, classe_test = train_test_split(previsores, classe)

from sklearn.model_selection import MLPClassifier
regressor = MLPClassifier(verbose = True, 
                          max_iter=5000, tol= 0.000010, 
                          solver='sgd', hidden_layer_sizes=(3939), 
                          activartion='relu') #this is hidden_layer is 3939, because we have 13 input colunms and 303 lines
regressor.fit(previsores_training, classe_training)
previsores_training = regressor.predict(previsores_test) 

#Results with acuracy and metrics
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

