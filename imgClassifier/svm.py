import os
import time

import joblib
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from fetchedData import X, y

"""
ordinary SVM
"""

# create a svm Classifier
ordinaryClf = svm.SVC(kernel='linear', C=1, random_state=42)  # Linear Kernel

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

# training the model
ordinaryClf.fit(X_train, y_train)

# testing the model
y_pred = ordinaryClf.predict(X_test)

# a look at the statistics
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))

"""
SVM with 10-way cross validation.

TODO you should choose by yourself which technique to apply 
(for the splitting technique) and not the default one
"""

# create a svm Classifier
clf = svm.SVC(kernel='linear', C=1, random_state=42)  # Linear Kernel

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

# train the model with 10 way cross validation
scores = cross_val_score(clf, X_train, y_train, cv=10)

# a look at the statistics
print("mean:", end=" ")
print(scores.mean())

print("sd:", end=" ")
print(scores.std())

# testing the model
y_pred = clf.predict(X_test)

# a look at the statistics
print("accuracy: ", metrics.accuracy_score(y_test, y_pred))

"""
LINEAR RIGHT WAY 
"""

svmClf = svm.SVC(random_state=42)

"""
parameter C: For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a 
better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the 
optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.
"""

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

properties = {
    "C": [0.02, 0.12, 0.49, 0.90, 7, 20, 50, 100],
    "kernel": ["linear"]
}

start_time = time.time()

# instantiate the tuning process
tuned_svcclsf_lin = GridSearchCV(
    svmClf, properties, scoring="accuracy", cv=10, return_train_score=True, verbose=6, n_jobs=10
)

# start the tuning process
tuned_svcclsf_lin.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
print("Best Score: {:.3f}".format(tuned_svcclsf_lin.best_score_))
print("Best Params: ", tuned_svcclsf_lin.best_params_)

# save results
if not os.path.exists('../tuned_models'):
    os.mkdir('../tuned_models')

# to load previously saved results
joblib.dump(tuned_svcclsf_lin, '../tuned_models/svc_linear_tuning_results.pkl')

# just statistical graphs
print(tuned_svcclsf_lin.cv_results_)
tuned_svcclsf_lin_results = pd.DataFrame(tuned_svcclsf_lin.cv_results_)

table = tuned_svcclsf_lin_results[["params", "mean_test_score", "mean_train_score"]]
print(table)

sns.set(rc={"figure.figsize": (12, 8)})
sns.lineplot(data=tuned_svcclsf_lin_results, x="param_C", y="mean_test_score")

# print the best estimator and the best score
print(tuned_svcclsf_lin.best_estimator_)
print(tuned_svcclsf_lin.best_score_)

test_acc = f1_score(y_true=y_train,
                    y_pred=tuned_svcclsf_lin.predict(X_train),
                    average='weighted')

print("Train F1 Score on original dataset: {}".format(test_acc))

# best parameters from automatic parameters tuning
svc_lin_clsf = svm.SVC(**tuned_svcclsf_lin.best_params_)

# train the new model
svc_lin_clsf.fit(X_train, y_train)

# prediction on the training set
svc_lin_train_pred = svc_lin_clsf.predict(X_train)
print("Training:")
print('F1 Score: {}'.format(f1_score(y_true=y_train, y_pred=svc_lin_train_pred, average='weighted')))

# prediction on them test set
svc_lin_test_pred = svc_lin_clsf.predict(X_test)
print("Testing:")
print('F1 Score: {}'.format(f1_score(y_true=y_test, y_pred=svc_lin_test_pred, average='weighted')))

"""
POLYNOMIAL OF DEGREE 2 RIGHT WAY 
"""

# automatic parameters tuning
svcclsf_pol = svm.SVC(random_state=42)

properties = {
    "C": [0.02, 0.12, 0.49, 0.90, 7, 20, 50, 100],  # soft to hard margin
    "kernel": ["poly"],
    "degree": [2],
    "gamma": ["auto", 0.1, 1]
}

# split the data between training and testing data (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

start_time = time.time()

tuned_svcclsf_pol = GridSearchCV(
    svcclsf_pol, properties, scoring="accuracy", cv=10, return_train_score=True, verbose=6, n_jobs=10
)

tuned_svcclsf_pol.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
print("Best Score: {:.3f}".format(tuned_svcclsf_pol.best_score_))
print("Best Params: ", tuned_svcclsf_pol.best_params_)

# save results
if not os.path.exists('../tuned_models'):
    os.mkdir('../tuned_models')

joblib.dump(tuned_svcclsf_pol, '../tuned_models/svc_polynomial_tuning_results.pkl')

# to load previously saved results
tuned_svcclsf_pol = joblib.load("../tuned_models/svc_polynomial_tuning_results.pkl")

# just printing some graphs
print(tuned_svcclsf_pol.cv_results_)
tuned_svcclsf_pol_results = pd.DataFrame(tuned_svcclsf_pol.cv_results_)

table = tuned_svcclsf_pol_results[["params", "mean_test_score", "mean_train_score"]]
print(table)

sns.set(rc={"figure.figsize": (12, 8)})
sns.lineplot(data=tuned_svcclsf_pol_results, x="param_C", y="mean_test_score")

# print the best estimator and the best score
print(tuned_svcclsf_pol.best_estimator_)
print(tuned_svcclsf_pol.best_score_)

test_acc = f1_score(y_true=y_train,
                    y_pred=tuned_svcclsf_pol.predict(X_train),
                    average='weighted')
print("Train F1 Score on original dataset: {}".format(test_acc))

# best parameters from automatic parameters tuning
svc_pol_clsf = svm.SVC(**tuned_svcclsf_pol.best_params_)

# train the new model
svc_pol_clsf.fit(X_train, y_train)

# check the accuracy for the training dataset
svc_pol_train_pred = svc_pol_clsf.predict(X_train)
print("Training:")
print('F1 Score: {}'.format(f1_score(y_true=y_train,
                                     y_pred=svc_pol_train_pred,
                                     average='weighted')))

# check the accuracy for the testing dataset
svc_pol_test_pred = svc_pol_clsf.predict(X_test)
print("Testing:")
print('F1 Score: {}'.format(f1_score(y_true=y_test,
                                     y_pred=svc_pol_test_pred,
                                     average='weighted')))

"""
RBF KERNEL RIGHT WAY
"""

# automatic parameters tuning
svcclsf_RBF = svm.SVC(random_state=42)

properties = {
    "C": [0.02, 0.12, 0.49, 0.90, 7, 20, 50, 100],  # soft to hard margin
    "kernel": ["rbf"],
    "gamma": ["auto", 0.1, 1]
}

start_time = time.time()

# initialize the tuning process
tuned_svcclsf_rbf = GridSearchCV(svcclsf_RBF, properties, scoring="accuracy", cv=10, return_train_score=True,
                                 verbose=6, n_jobs=10)
# start the tuning process
tuned_svcclsf_rbf.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

print("Best Score: {:.3f}".format(tuned_svcclsf_rbf.best_score_))
print("Best Params: ", tuned_svcclsf_rbf.best_params_)

# to load previously saved results
tuned_svcclsf_rbf = joblib.load("../tuned_models/svc_rbf_tuning_results.pkl")

print(tuned_svcclsf_rbf.cv_results_)
tuned_svcclsf_rbf_results = pd.DataFrame(tuned_svcclsf_rbf.cv_results_)

print(tuned_svcclsf_rbf_results[["params", "mean_test_score", "mean_train_score"]])

sns.set(rc={"figure.figsize": (12, 8)})
sns.lineplot(data=tuned_svcclsf_rbf_results, x="param_C", y="mean_test_score")

print(tuned_svcclsf_rbf.best_estimator_)
print(tuned_svcclsf_rbf.best_score_)

test_acc = f1_score(y_true=y_train,
                    y_pred=tuned_svcclsf_rbf.predict(X_train),
                    average='weighted')

print("Train F1 Score on original dataset: {}".format(test_acc))

# best parameters from automatic parameters tuning
svc_rbf_clsf = svm.SVC(**tuned_svcclsf_rbf.best_params_)

svc_rbf_clsf.fit(X_train, y_train)

svc_rbf_train_pred = svc_rbf_clsf.predict(X_train)
svc_rbf_test_pred = svc_rbf_clsf.predict(X_test)

print("Training:")
print('F1 Score: {}'.format(f1_score(y_true=y_train, y_pred=svc_rbf_train_pred, average='weighted')))

print("Testing:")
print('F1 Score: {}'.format(f1_score(y_true=y_test,
                                     y_pred=svc_rbf_test_pred,
                                     average='weighted')))
