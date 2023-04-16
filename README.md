# DecisionTreeS-Project
MACHINE LEARNING


------------------- Making a prediction of GENRE type od song a user would probable listen or play base on past histories --------------------------


import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library
music_data = pd.read_csv('music.csv')
music_data

## split data by dropping the colunms 

X = music_data.drop(columns=['genre'])
X

## Split to Output data ( is the prediction for the answer , training to make predictions for the model)

y = music_data['genre']
y

#  Build a Model; Using Decision Tree as the Machine algorithm ( from Sklearn library)

from sklearn.tree import DecisionTreeClassifier

## create an object and set to an instance - the model
model = DecisionTreeClassifier()
## train it to learn patterns in the data
model.fit(X, y)
## now make prediction of a 21 year old male and 22 years female music they likes
predictions = model.predict([[21, 1], [22, 0]])
predictions ## and click enter on the variable

# Measure accuracy of Model ( splitting the model into Training and Testing )

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## create an object and set to an instance - the model
model = DecisionTreeClassifier()
#allocate 80% for training and 20% for testing to calculate the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## X_train, X_test are the input set while 
## y_train, and y_test are the output set.

model.fit(X_train, y_train) ## passing only the train data set to evaluate the model
predictions = model.predict(X_test) ## X_test contains the input values for testing

## To get the accurcy, add a library for this
## and compare prediction by the actual values we have for testing

score = accuracy_score(y_test, predictions) 
## y_test is the expected values and prediction contains actual values
score

# Model Persistence ( no need to train a model if you have a new users, only by loading already trained model from a file )
#To save a trained model for future model testing


import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library

## as metohd for saving and loading model use this for model persistency
import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

## create an object and set to an instance - the model
model = DecisionTreeClassifier()
## train it to learn patterns in the data
model.fit(X, y)


joblib.dump(model, 'music-recommender.joblib') ## save and laoding


## now make prediction of a 21 year old male and 22 years female music they likes
#predictions = model.predict([[21, 1], [22, 0]])
#predictions ## and click enter on the variable

## Model Persistence ( TO LOAD AN EXISTING MODEL )

import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library
## as metohd for saving and loading model use this for model persistency
import joblib
## laoding an exisiting model an make prediction with it

model = joblib.load('music-recommender.joblib') 

## now make prediction of a 21 year old male and 22 years female music they likes
predictions = model.predict([[21, 1], [22, 0]])
predictions ## and click enter on the variable

# VISUALISING THE MODEL PREDICTION

import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

## create an object and set to an instance - the model
model = DecisionTreeClassifier()

model.fit(X, y)

tree.export_graphviz(model, out_file = 'music-recommender.dot',
                    feature_names=['age', 'gender'],
                    class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
