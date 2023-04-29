import numpy as np
import pandas as pd

data = pd.DataFrame({"x0": [1, 1, 1, 1], "x1": [0, 0, 1, 1], "x2": [
                    0, 1, 0, 1], "y": [0.0, 1.5, 2.0, 2.5]})  # x0 is all zeros
theta = [0, 0, 0]  # initialize theta with zeros


def getSumSquareErrors(data, theta):
    '''
    Returns the sum of squared errors by comparing y to the dot product of
    '''
    SSE = 0

    for index, row in data.iterrows():
        y_hat = row["x0"]*theta[0] + row["x1"] * \
            theta[1] + row["x2"]*theta[2]
        y = row["y"]
        SSE += (y - y_hat)**2
    return SSE


def adjustTheta(theta, data, alpha, sampleCount):
    # update theta[0]
    sumErrors = 0
    for index, row in data.iterrows():
        x = row.values.tolist()[:-1]  # exclude "y" column
        y = row.values[-1]
        y_hat = np.dot(x, theta)
        # print("y_hat = " + str(y_hat) + ", y = " + str(y))
        sumErrors += (y_hat-y)
    theta[0] = theta[0] - alpha*sumErrors/sampleCount

    # updated theta[1-2]
    sumErrorsX = [0, 0, 0]
    for m in range(1, 3):
        for index, row in data.iterrows():
            x = row.values.tolist()[:-1]  # exclude "y" column
            y = row.values[-1]
            y_hat = np.dot(x, theta)
            sumErrorsX[m] += (y_hat-y)*x[m]

        theta[m] = theta[m] - alpha*sumErrorsX[m]/sampleCount


# Run the algorithm 100 times to determine theta
iterations = 15
alpha = 0.1
sampleCount = len(data)
for iter in range(iterations):
    SSE = getSumSquareErrors(data, theta)
    print(SSE)
    adjustTheta(theta, data, alpha, sampleCount)
    # print(theta)

print(theta)
