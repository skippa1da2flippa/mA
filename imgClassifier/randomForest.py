import os

import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from fetchedData import X, y

"""
TODO check n_estimators the higher the number the higher the accuracy of this machine learning model but the longer it 
takes to be trained
"""

"""
ordinary Random forest 
"""

# create a random forest classifier
ordinaryClf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

# training the model
ordinaryClf.fit(X_train, y_train)

# testing the model
y_pred = ordinaryClf.predict(X_test)

# a look at the statistics
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))

"""
Random forest with 10-way cross validation
"""

"""
Parameters tuning 
"""

properties = {
    "n_estimators": [x for x in range(20, 140, 20)],
    "max_depth": [x for x in range(20, 120, 10)],
    "min_samples_split": [x for x in range(2, 20, 1)],
    "min_samples_leaf": [x for x in range(2, 20, 1)],
}


# Class used to represent tuning data
class ParametersWrapper:

    def __int__(self, n_estimators: int = 0, max_depth: int = 0, min_samples_split: int = 0, min_samples_leaf: int = 0):
        self.n_estimator = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf


classifier = RandomForestClassifier()

# instantiate the tuning object
tunedParams = GridSearchCV(
    classifier, properties, scoring="accuracy", cv=2, return_train_score=True, verbose=4
)

# start the tuning process
tunedParams.fit(X_train, y_train)

# a look at the statistics
print("Best Score: {:.3f}".format(tunedParams.best_score_))
print("Best Params: ", tunedParams.best_params_)
print(tunedParams.best_estimator_)
print(tunedParams.best_score_)

# create a new classifier with the tuned parameters
clf = RandomForestClassifier(n_estimators=2, max_depth=None, min_samples_split=2, random_state=0)

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

# train the model with 10 way cross validation
scores = cross_val_score(clf, X_train, y_train, cv=10)

# a look at the statistics
print("mean", scores.mean())

print("sd", scores.std())

# testing the model
y_pred = clf.predict(X_test)

# a look at the statistics
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))

"""
    RIGHT WAY
"""

# instantiate a new random forest classifier
classifier = RandomForestClassifier()

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

# object containing all the possible values for each hyperparameter
properties = {
    "n_estimators": [x for x in range(20, 140, 20)],
    "max_depth": [x for x in range(20, 120, 20)],
    "min_samples_split": [x for x in range(2, 10, 2)]
}

# instantiate the tuning object
tunedParams = GridSearchCV(
    classifier, properties, scoring="accuracy", cv=10, return_train_score=True, verbose=4, n_jobs=10
)

# start the tuning process
tunedParams.fit(X_train, y_train)

# save results
if not os.path.exists('../tuned_models'):
    os.mkdir('../tuned_models')

# to load previously saved results
joblib.dump(tunedParams, '../tuned_models/randomForest.pkl')

# a look at the statistics
print("Best Score: {:.3f}".format(tunedParams.best_score_))
print("Best Params: ", tunedParams.best_params_)
print(tunedParams.best_estimator_)
print(tunedParams.best_score_)

# create a new classifier with the best hyperparameters
bestClassifier = RandomForestClassifier(**tunedParams.best_params_)

# training the model
bestClassifier.fit(X_train, y_train)

# testing the model
y_pred = bestClassifier.predict(X_test)

# print the accuracy
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))
