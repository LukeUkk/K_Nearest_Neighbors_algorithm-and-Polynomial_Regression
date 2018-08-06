# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:38:56 2018

@author: Luke
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns #keep


data = pd.read_csv('\\UNI\\Year 2\\Algorithms for data mining\\assingment 2\\Assignment item 2 Dataset.csv')

#describea=pd.DataFrame.describe(a) #Test function

#!!conver the class to 0.1 vector

pick = input ("Do you want to Run: A) ANN KNN along with the accuracy and CV accuracy. B) Graph plots. [A/B]? : ")
if pick == "A":
    y = data['Class'] #used to split and assign value y
    x = data.drop('Class', axis=1) #used to split and assign value to x
    
    X_train, X_test, y_train, y_test = train_test_split(x, y) #take x and y and then use the train test split to preform a split on the data 

    mlp = MLPClassifier(hidden_layer_sizes=(50),max_iter=200,activation=('logistic'))
    mlp.fit(X_train,y_train)

    predictions = mlp.predict(X_test)

    print('Confusion_Matrix')
    print(confusion_matrix(y_test,predictions))
    print('Classification_report')
    print(classification_report(y_test,predictions))
    
    
    scores = cross_val_score(mlp, X_train, y_train, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('---------------^Cross Validation for ANN^---------------')
    
    k = KNeighborsClassifier(n_neighbors=10) #Classifier implementing the k-nearest neighbors vote.
    kfit = k.fit(X_train,y_train) #Fit the model using X as training data and y as target values
    
    kpredict = k.predict(X_test) #Predict the class labels for the provided data
    
    print('---------------This is the accuracy result for this run of k nearest neighbors---------------')
    print(accuracy_score(y_test,kpredict)) #In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print(confusion_matrix(y_test,kpredict)) # Data that is outputs can be used to figure out the accuracy.
    
    kscores = cross_val_score(k, X_train, y_train, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (kscores.mean(), kscores.std() * 2))
    print('---------------^Cross Validation for KNN^---------------')

#------------- Graphs -------------#
if pick == "B":
    pick_graph = input ("Which Graph do you want to Run: A) Box_Plot. B) Distplot. [A/B]? : ")
    if pick_graph == "A":
        data.boxplot(column='PressureA', by='Class') #w works dont deletle
    elif pick_graph == "B":
        x = data.iloc[:100,10] # This code shows python where to locate and split the data at the desired lenght represented by the numbers this now makes x hold the diabetic class data
        y = data.iloc[-100:200,10]  # This shows python to locate and split the data at the inputed leanght This is Fro the Diabetic retinopathy data
        
        sns.distplot(y,label='DR')
        sns.distplot(x,label='Diabetes')
#--------------------------
