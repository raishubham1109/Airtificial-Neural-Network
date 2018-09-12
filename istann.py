# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 21:51:51 2018

@author: Shubham Rai
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:,2] =labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X= X[:,1:]# removing ist column 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# importing library keras sequential and dense
import keras
from keras.models import Sequential
from keras.layers import Dense

#intialising ANN
classifire = Sequential()
# adding the input layer and first hidden  layer to ANN
classifire.add(Dense(output_dim = 6, init = 'uniform',activation="relu", input_dim=11))
# adding 2nd hidden  layer to ANN

classifire.add(Dense(output_dim = 6, init = 'uniform',activation="relu"))

#adding the output layer

classifire.add(Dense(output_dim = 1, init = 'uniform',activation="sigmoid"))
#Compile the modle
#Loss is category_crosswntropy for multipe output categorey
classifire.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])
#Fitting the ANN to the trainig set

classifire.fit(X_train,y_train, batch_size=10, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifire.predict(X_test)
y_pred= (y_pred>0.5)
# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new = classifire.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new = (new> 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
