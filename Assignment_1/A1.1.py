import pandas as pd
import numpy as np
import math
import pprint


# Recursive function that returns a node with subnodes (i.e., a tree)


def createDecisionTree(data, depth=0):
    '''
    Recursively creates a decision tree of nodes based on a data set.
    Assumes the last column in the dataset is the dependent variable/label.
    Each node will look like this:
        node: {
            _values: {},
            featureName: xxxxx,
            isLeaf: False/True,
            entropy: x,
            informationGain: x,
            splitVal: x,
            leftNode: {},
            rightNode: {},
            prediction: x, (if isLeaf == True)
        }
    '''
    # 0. instantiate this node
    node = {}
    # get the counts of each label value
    node["_values"] = dict(data["class"].value_counts())
    entropy = getEntropy(data)
    node['entropy'] = entropy
    node["isLeaf"] = False

    # Base case 1 - stop when tree height = 3
    if depth + 1 > 3:
        node["isLeaf"] = True

    # Base case 2:  if entropy < 0.1 -> this is a Leaf node (no more splits required)
    if entropy < 0.1:
        node["isLeaf"] = True

    # Base case 3: there are no features (data contains only the "class" column)
    if len(list(data.columns)) == 1:
        node["isLeaf"] = True

    # Base case 4: If there are < 3 rows -> use mode()[0]
    if len(data) < 3:
        node["isLeaf"] = True

    #  If leaf node == True, make a prediction and return the node
    if node['isLeaf'] == True:
        # predict the most frequntly occuring value
        node['prediction'] = data["class"].mode().tolist()[0]
        return node

    # 2. Look at all features and decide the best way to split the data into two subgroups
    bestSplit = getBestSplit(data)
    node["featureName"] = bestSplit["featureName"]
    node["informationGain"] = getInformationGain(data, bestSplit)
    node["splitVal"] = bestSplit["splitVal"]

    # 3. Create each branch using the data from each split

    # 3a. Create Left Branch
    leftNodeData = data.loc[data[bestSplit["featureName"]]
                            < bestSplit["splitVal"]]  # filter: only data < splitVal for left branch

    # Optional... allows you to split on a feature only once
    # leftNodeData = leftNodeData.drop(node["featureName"], axis=1)

    node["leftNode"] = createDecisionTree(leftNodeData, depth+1)

    # 3b. Right branch
    rightNodeData = data.loc[data[bestSplit["featureName"]]
                             >= bestSplit["splitVal"]]  # filter: only data >= splitVal for right branch

    # Optional... allows you to split on a feature only once
    # rightNodeData = rightNodeData.drop(node["featureName"], axis=1)

    node["rightNode"] = createDecisionTree(rightNodeData, depth+1)

    return node


def getEntropy(data):
    '''
    Returns the entropy (between 0 and 1) of the data. Input "data" is a dataframe.
    '''
    # Get the number of unique output values
    uniqueVals = data.iloc[:, -1].value_counts()

    # For each unique output value, calculate the entropy and sum them
    if len(uniqueVals) == 1:  # there is only one "class" value -> entropy = 0
        return 0
    entropy = 0  # initialize entropy to 0
    for val in uniqueVals.index.tolist():
        # get the probability "p" by dividing count of "val" by total count
        p = uniqueVals[val] / len(data)
        # add weighted entropy fraction to entropy
        # use log base 2 since there are 2 "class" values
        entropy += -(p*math.log(p, 2))
    return entropy


def getBestSplit(data):
    '''
    Returns an object containing the best way to split the data:
    {   
     featureName: "xxxx",
     splitVal: x,
     expEntropy: x,
    }
    '''
    features = data.columns[:-1].tolist()  # get list of feature names
    # instantiate bestSplit as placeholder with first feature, first row of data
    bestSplit = {
        "featureName": features[0],  # first feature name as placeholder
        # first row of data as placeholder
        "splitVal": data[features[0]].tolist()[0],
        "expEntropy": getExpectedEntropy(data, features[0], data[features[0]].tolist()[0]),
    }
    # Find the actual bestSplit
    for feature in features:
        # find the best value to split the data which minimizes expected entropy
        expectedEntropy = 1  # start by assuming highest entropy
        for value in data[feature].tolist():  # for every value for the target feature
            # get expected entropy
            EE = getExpectedEntropy(data, feature, value)
            if EE < expectedEntropy:  # update bestSplit if EE is less than expectedEntropy
                bestSplit["featureName"] = feature
                bestSplit["splitVal"] = value
                bestSplit["expEntropy"] = EE
                expectedEntropy = EE
    return bestSplit


def getExpectedEntropy(data, featureName, value):
    '''
    Returns the expected entropy if the data is split by featureName into 2 groups (1 < value, 2 >= value)
    '''

    dataset1 = data.loc[data[featureName] < value]
    dataset2 = data.loc[data[featureName] >= value]
    weight1 = len(dataset1) / len(data)
    weight2 = len(dataset2) / len(data)
    if weight1 == 0:  # if weight1 = 0, there are no rows where the feature value < the parameter "value"
        entropy2 = getEntropy(dataset2)
        return entropy2

    if weight2 == 0:  # if weight2 = 0, there are no rows where the feature value >= the parameter "value"
        entropy1 = getEntropy(dataset1)
        return entropy1

    entropy1 = getEntropy(dataset1)
    entropy2 = getEntropy(dataset2)
    return entropy1*weight1 + entropy2*weight2  # return weighted entropy


def getInformationGain(data, feature):
    '''
    Returns the information gain of the data if split by the parameter "feature".
    Assume the parameter "feature" is an object that has a key titled "expEntropy" (expected entropy
    for that feature must already be calculated).
    '''
    expectedEntropy = feature["expEntropy"]
    entropy = getEntropy(data)
    infoGain = entropy - expectedEntropy
    return infoGain


def printConciseTree(tree, indent):
    """
    Function to print the tree concisely (without all node keys) using pre-order traversal.
    """
    if tree["isLeaf"] == True:
        print("Prediction = " + str(tree["prediction"]))

    else:
        print("X " + str(tree["featureName"]), "<",
              tree["splitVal"], "?", tree["informationGain"])
        print("%sleft:" % (indent), end=" ")
        printConciseTree(tree["leftNode"], indent + indent)
        print("%sright" % (indent), end=" ")
        printConciseTree(tree["rightNode"], indent + indent)


def printFullTree(tree):
    '''
    Prints the decision tree in full (shows all node key/value pairs)
    '''
    pprint.pprint(tree)


def getIndex(featureName):
    if featureName == "feature1":
        return 0
    if featureName == "feature2":
        return 1
    if featureName == "feature3":
        return 2
    if featureName == "feature4":
        return 3


def predictClass(x, tree):
    '''
    Recursive function to predict the "class" (dependent variable) based on
    one observation containing 4 independent features.
    '''
    if tree["isLeaf"] == True:
        return tree["prediction"]
    else:
        # Get feature name at top of tree
        splitIndex = getIndex(tree["featureName"])
        # If the value is less than split index -> go left, else go right
        if x[splitIndex] < tree["splitVal"]:
            return predictClass(x, tree["leftNode"])
        else:
            return predictClass(x, tree['rightNode'])


def predictDataSet(tree, dataset):
    '''
    Returns a list of predictions besed on an input dataset. "dataset" is a
    matrix of n observations, each with 4 independent features
    '''
    predictions = []  # initialize predictions list
    # For each row, add a prediction to "predictions"
    for rowIndex in range(len(dataset)):
        predictions.append(predictClass(dataset.iloc[rowIndex], tree))
    return predictions


def getTestingAccuracy(tree, dataset):
    '''
    Returns the accuracy of the tree based on a set of independent features (X).
    1. Calculate a predicted Y for each instance of X
    2. Determine the % of correct Y values vs. % of wrong Y values
    '''
    right = 0
    total = 0
    predictions = predictDataSet(tree, dataset)
    for n in range(len(predictions)):
        if predictions[n] == dataset.iloc[n, dataset.columns.get_loc("class")]:
            right += 1
        total += 1
    return float(right/total)


def getTrainingTestData(data, testPercent):
    '''
    Returns the training and test data.
    '''
    from sklearn.model_selection import train_test_split
    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=testPercent)
    test_data = pd.concat([x_test, y_test], axis=1)
    train_data = pd.concat([x_train, y_train], axis=1)
    return test_data, train_data


if __name__ == "__main__":
    # Get data from CSV
    data = pd.read_csv("data.csv")

    # Split data into training and test data
    testPercent = 0.2
    testData, trainData = getTrainingTestData(data, testPercent)

    # Create and print tree
    tree = createDecisionTree(trainData)
    printConciseTree(tree, "--")

    accuracy = getTestingAccuracy(tree, testData)
    print(accuracy)
