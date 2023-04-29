'''
# Random Forest classifier 
1. Load iris data set.

Investigate following parameters of Random Forest classifier and tune them using Randomized Search and Grid Search. 
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt','log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 1000,10)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 8, 11,14]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4,6,8]


2. Use seed 1 to split data in 80-20 train-test configuration.  Train a Random Forest classifier 
with each unique configuration and record train/test accuracy, precision and recall in the results 
dataframe. This dataframe will have 5 columns (each corresponding to tuning parameter) and each row 
will correspond to each unique configuration. 5x5x5x5x5 rows. Analyse of the impact of each tuning 
parameter on predictor performance. **(15 Points)**

3. From the results of the above find the best estimators and use them for classifcation once again and 
evaluate the performance using 10 fold cross validation. **(15 Points)**
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from pprint import pprint


# return the accruacy, precision, and recall for a given set of inputs
def getAccuracy(n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, X_train,
                X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                random_state=123456)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    return accuracy, recall, precision


# Load data
iris = datasets.load_iris()

# Make Dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# split data
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target,
                                                    test_size=0.2, stratify=iris.target, random_state=123456)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000, 10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 8, 11, 14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# RANDOM SEARCH
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, cv=10, n_iter=100,
                               n_jobs=-1, random_state=123456)
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)
