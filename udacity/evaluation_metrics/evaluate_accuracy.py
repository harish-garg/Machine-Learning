#
# In this and the following exercises, you'll be adding train test splits to the data
# to see how it changes the performance of each classifier
#
# The code provided will load the Titanic dataset like you did in project 0, then train
# a decision tree (the method you used in your project) and a Bayesian classifier (as
# discussed in the introduction videos). You don't need to worry about how these work for
# now.
#
# What you do need to do is import a train/test split, train the classifiers on the
# training data, and store the resulting accuracy scores in the dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')
# Limit to numeric data
X = X._get_numeric_data()
# Separate the labels
y = X['Survived']
# Remove labels from the inputs, and age due to missing data
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

# The decision tree classifier
clf1 = DecisionTreeClassifier()
clf1.fit(X,y)
print "Decision Tree has accuracy: ",accuracy_score(clf1.predict(X),y)
# The naive Bayes classifier

clf2 = GaussianNB()
clf2.fit(X,y)
print "GaussianNB has accuracy: ",accuracy_score(clf2.predict(X),y)

answer = {
 "Naive Bayes Score": 0,
 "Decision Tree Score": 0
}
