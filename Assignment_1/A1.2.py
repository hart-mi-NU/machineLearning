'''
Use the Decision Tree Classifier from the Sklearn Library and use gini index as a splitting measure. Use the data.csv dataset.
Calculate accuracy for this model. 
Print the Decision tree and compare the Decision Trees generated from your code and Sklearn.
'''

import pandas as pd
import numpy as np
import math
import pprint
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz


def createDecisionTree(data, maxDepth):
    dTree = tree.DecisionTreeClassifier(
        max_depth=maxDepth, criterion="gini")
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    dTree = dTree.fit(X, Y)
    return dTree


def printTree(decisionTree):
    r = tree.export_text(decisionTree, feature_names=[
        'feature1', 'feature2', 'feature3', 'feature4'])
    print(r)


def getAccuracy(decisionTree, x_test, y_test):
    '''
    Returns the accuracy of the sklearn decision tree.
    '''
    predcitions = decisionTree.predict(x_test)
    accuracy = accuracy_score(y_test, predcitions)
    return accuracy


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    dTree = createDecisionTree(data, 3)
    printTree(dTree)
    accuracy = getAccuracy(dTree, data.iloc[:, :-1], data.iloc[:, -1])
    print(accuracy)
