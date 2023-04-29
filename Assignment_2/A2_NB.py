# Display and explain the likelihood (Class conditional probabilities) for each input variable.
import pprint
import numpy as np
import pandas as pd


# Returns a dictionary with the probabilities of each class
def getClassProbabilities(data):
    # Get the "clazz" column
    data = data.loc[:, "clazz"]
    output = {}
    # Get the probabilities
    for key, val in dict(data.value_counts(dropna=False, normalize=True)).items():
        output[key] = val
    # Get  the counts
    # for key, val in dict(data.value_counts(dropna=False)).items():
    #     output[key]["count"] = val
    return output


# Returns a dictionary with the probabilities of each input variable
def getInputProbabilities(data):
    # exclude the "clazz" column
    data = data.iloc[:, :-1]
    columns = data.columns.tolist()
    output = {}
    for col in columns:
        output[col] = {}
        for key, val in dict(data[col].value_counts(dropna=False, normalize=True)).items():
            output[col][key] = val
        # for key, val in dict(data[col].value_counts(dropna=False)).items():
        #     output[col][key]["count"]: val
    return output


# Returns a dictionary with the joint probabilities of every input variable and class combination
def getJointProbabilities(data):
    op = data.loc[:, "clazz"]  # last column is the op
    ip = data.iloc[:, :-1]  # first 6 columns are the ip
    output = {}
    # For each class in "clazz" column

    for className, classProb in dict(op.value_counts(dropna=False, normalize=True)).items():
        # filter the data by the given class
        ip = data.loc[data["clazz"] == className].iloc[:, :-1]
        # print(str(className) + " length = " + str(len(ip)))
        output[className] = {}
        # For each column in ip
        for ipCol in ip.columns.tolist():
            check = 0
            output[className][ipCol] = {}
            # Add all joint probabilities
            # For each value in ipCol
            for key, val in dict(ip[ipCol].value_counts(normalize=True)).items():
                p = round(val*classProb, 5)
                output[className][ipCol][key] = p
                check += p
            # print("col: " + str(ipCol) + ", check = " + str(check))
    return output


# Calculate class-conditional likelihoods for each input variable
def getClassConditionalLikelihoods(classProbabilities, jointProbabilities):
    output = {}
    # For each clazz
    for clazz in classProbabilities.keys():
        # Store P(class)
        pClass = classProbabilities[clazz]
        # print(pClass)
        # Iterate through every joint probability and divide it by pClass, then store in output
        output[clazz] = {}
        for j_column in jointProbabilities[clazz].keys():
            output[clazz][j_column] = {}
            for colOption in jointProbabilities[clazz][j_column].keys():
                jProb = jointProbabilities[clazz][j_column][colOption]
                p = jProb / pClass
                # print("jProb of class " + str(clazz))
                output[clazz][j_column][colOption] = p
    return output


# Display the class conditional probabilities for each input variable
def printClassCondProbs(classCondProbs):
    for _class in classCondProbs:
        print("class: " + str(_class))
        for _colName in classCondProbs[_class]:
            print("   " + str(_colName))
            for _variable in classCondProbs[_class][_colName]:
                prob = classCondProbs[_class][_colName][_variable]
                print("      " + str(_variable) + ": " + str(round(prob, 4)))


# Function to classify an instance (posterior)
def classifyInstance(testInstance, classCondProbs, classProbs):
    # Ensure the testInstance doesn't include the "class" column
    data = testInstance.drop(index=["clazz"])
    # Find the max likelihood
    max = 0
    prediction = None
    # For each class
    for className, classP in classProbs.items():
        prob = 1  # initialize the probability at 1
        # Find each class conditional probability
        for ipVar in data.keys().tolist():
            value = data[ipVar]
            # if value is in training data -> use it
            if value in classCondProbs[className][ipVar].keys():
                likelihood = classCondProbs[className][ipVar][value]
            else:  # else assume likelihood is 0
                likelihood = 0
            prob *= likelihood  # get product of all likelihoods
        prob *= classP  # multiply classP by product of all likelihoods
        # update max
        if prob > max:
            prediction = className
            max = prob
    return prediction


# Function to classify all instanes given a matrix/vector of test data
def classifyTestData(testData, classCondProbs, classProbs):
    output = testData.copy()
    for index, row in testData.iterrows():
        predictedClass = classifyInstance(row, classCondProbs, classProbs)
        output.at[index, "clazz"] = predictedClass
    return output


# Function to determine the accuracy of the model
def getModelAccuracy(testDataActual, testDataFromModel):
    total = 0
    right = 0
    for index, row in testDataActual.reset_index().iterrows():
        total += 1
        actual = testDataActual.iloc[index, -1]
        predicted = testDataFromModel.iloc[index, -1]
        if actual == predicted:
            right += 1
    return round(right / total, 4)


dataset = pd.read_csv('car-eval.csv')
test_indis = 2
test_dataset = dataset[dataset.index % test_indis == 0]
train_dataset = dataset[dataset.index % test_indis != 0]

classProbs = getClassProbabilities(train_dataset)
# inputProbs = getInputProbabilities(train_dataset)
jointProbs = getJointProbabilities(train_dataset)
classCondProbs = getClassConditionalLikelihoods(classProbs, jointProbs)
printClassCondProbs(classCondProbs)
dataPredictions = classifyTestData(test_dataset, classCondProbs, classProbs)
modelAccuracy = getModelAccuracy(test_dataset, dataPredictions)
print(modelAccuracy)
