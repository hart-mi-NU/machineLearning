from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

n = 2  # dimensionality
Ntrain = 1000
Ntest = 5000
ClassPriors = [0.5, 0.5]
r0 = 2
r1 = 4
sigma = 1
mu = 0


def generate_data(N):
    data_labels = np.random.choice(2, N, replace=True, p=ClassPriors)
    ind0 = np.array((data_labels == 0).nonzero())
    ind1 = np.array((data_labels == 1).nonzero())
    N0 = np.shape(ind0)[1]
    N1 = np.shape(ind1)[1]
    theta0 = np.random.uniform(-np.pi, np.pi, size=N0)
    theta1 = np.random.uniform(-np.pi, np.pi, size=N1)
    x0 = mu + sigma * \
        np.random.standard_normal((N0, n)) + r0 * \
        np.transpose([np.cos(theta0), np.sin(theta0)])
    x1 = mu + sigma * \
        np.random.standard_normal((N1, n)) + r1 * \
        np.transpose([np.cos(theta1), np.sin(theta1)])
    data_features = np.zeros((N, 2))
    np.put_along_axis(data_features, np.transpose(ind0), x0, axis=0)
    np.put_along_axis(data_features, np.transpose(ind1), x1, axis=0)
    return (data_labels, data_features)


def plotScatter(x, y, title):

    fig = plt.figure(dpi=100)
    ax = fig.gca()  # projection='3d')
    labels = {0: "class0", 1: "class1"}
    ax.scatter(x[y == 0, 0], x[y == 0, 1], label="class0")
    ax.scatter(x[y == 1, 0], x[y == 1, 1], label="class1")
    if title:
        plt.title(title)
    else:
        plt.title("True Data Distribution: " + len(x) + " samples")
    plt.show()


# Generate Data
yTrain, xTrain = generate_data(1000)


# Show the data
# plotScatter(xTrain, yTrain, "Training Data")


# Create model and train it
model = SVC(kernel="rbf")
model.fit(xTrain, yTrain)

# View the default model parameters
# print(model.get_params())

# Get the Test data
yTest, xTest = generate_data(5000)

# GET THE MODEL ERROR
error = 1 - model.score(xTest, yTest)
print(f"SVM accuracy = {(1 - error):.4f}")


C_range = np.logspace(-2, 10, 10)
gamma_range = np.logspace(-9, 3, 10).tolist()+['scale', 'auto']
param_grid = dict(gamma=gamma_range, C=C_range, kernel=["rbf"])
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scoring = ["accuracy"]
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    scoring=scoring,
                    refit='accuracy',
                    n_jobs=-1,
                    cv=kfold,
                    verbose=0)
grid.fit(xTrain, yTrain)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)
# # Hyperparameter tuning

# # Set the
# C_range = np.logspace(-1, 1, 3)
# gamma_range = np.logspace(-1, 1, 3)
# # Define the search space
# param_grid = {
#     # Regularization parameter.
#     "C": C_range,
#     # Kernel type
#     "kernel": ['rbf'],
#     # Gamma is the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#     "gamma": gamma_range.tolist()+['scale', 'auto']
# }
# scoring = ["accuracy"]

# # Set up the k-fold cross-validation
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# # TODO calculate and vizualize the accuracy using several other parameters
# # Model1 -> C=0.1, gamma=0.1
# model1 = SVC(kernel="rbf")
# params = {"C": 0.1, "gamma": 0.1}
# model1.set_params(**params)
# model1.fit(xTrain, yTrain)
# print("model1 accuracy = " + str(model1.score(xTest, yTest)))

# # Model2 -> C=1, gamma=1
# model2 = SVC(kernel="rbf")
# params = {"C": 1.0, "gamma": 1.0}
# model2.set_params(**params)
# model2.fit(xTrain, yTrain)
# print("model2 accuracy = " + str(model2.score(xTest, yTest)))

# # Model3 -> C=3, gamma=3
# model3 = SVC(kernel="rbf")
# params = {"C": 3, "gamma": 3}
# model3.set_params(**params)
# model3.fit(xTrain, yTrain)
# print("model3 accuracy = " + str(model3.score(xTest, yTest)))

# # Model4 -> C=10, gamma=10
# model4 = SVC(kernel="rbf")
# params = {"C": 10, "gamma": 10}
# model4.set_params(**params)
# model4.fit(xTrain, yTrain)
# print("model4 accuracy = " + str(model4.score(xTest, yTest)))


# # GET THE BEST PARAMETERS
# # Define grid search
# grid_search = GridSearchCV(estimator=model,
#                            param_grid=param_grid,
#                            scoring=scoring,
#                            refit='accuracy',
#                            n_jobs=-1,
#                            cv=kfold,
#                            verbose=0)
# # Fit grid search
# grid_result = grid_search.fit(xTrain, yTrain)
# # Print grid search summary
# print("Best score = " + str(grid_result.best_score_))
# print("Best params = " + str(grid_result.best_params_))
# print("Best accuracy using Test Data = " +
#       str(grid_search.score(xTest, yTest)))


# MLP CLASSIFIER
# 100 neurons
mlp100 = MLPClassifier(random_state=1, max_iter=400).fit(xTrain, yTrain)
mlp100Score = mlp100.score(xTest, yTest)
print(f"MLP accuracy with 100 neurons = {mlp100Score:.4f}")
maxMLP = mlp100, mlp100Score

# 200 neurons
params = {
    "hidden_layer_sizes": (200, 200)
}
mlp200 = MLPClassifier(random_state=1, max_iter=400)
mlp200.set_params(**params)
mlp200.fit(xTrain, yTrain)
mlp200Score = mlp200.score(xTest, yTest)
print(f"MLP accuracy with 200 neurons = {mlp200Score:.4f}")
if mlp200Score > maxMLP[1]:
    maxMLP = mlp200, mlp200Score

# 25 neurons
mlp25 = MLPClassifier(random_state=1, max_iter=400)
params = {
    "hidden_layer_sizes": (25, 25)
}
mlp25.set_params(**params)
mlp25.fit(xTrain, yTrain)
mlp25Score = mlp25.score(xTest, yTest)
print(f"MLP accuracy with 25 neurons = {mlp25Score:.4f}")
if mlp25Score > maxMLP[1]:
    maxMLP = mlp25, mlp25Score

# 2 neurons
params = {
    "hidden_layer_sizes": (2,)
}
mlp2 = MLPClassifier(random_state=1, max_iter=400)
mlp2.set_params(**params)
mlp2.fit(xTrain, yTrain)
mlp2Score = mlp2.score(xTest, yTest)
print(f"MLP accuracy with 2 neurons = {mlp2Score:.4f}")
if mlp2Score > maxMLP[1]:
    maxMLP = mlp2, mlp2Score

bestLayerSize = maxMLP[0].get_params()["hidden_layer_sizes"]

print(
    f"Max accuracy is for {bestLayerSize} @ {maxMLP[1]}")


# create a mesh to plot in
h = 0.02
x_min, x_max = xTest[:, 0].min() - 1, xTest[:, 0].max() + 1
y_min, y_max = xTest[:, 1].min() - 1, xTest[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['100', '200', '25', '2']

# MAke the plots
for i, clf in enumerate((mlp100, mlp200, mlp25, mlp2)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(xTest[:, 0], xTest[:, 1], c=yTest,
                cmap=plt.cm.coolwarm, alpha=0.2)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
