import os
import time

import joblib
import pandas as pd
import sns
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score


from fetchedData import X, y, X_train, y_train
from imgClassifier.naiveBayes import predict, fit
from imgClassifier.test import KNN


def main():
    """
    # instantiate a new random forest classifier
    classifier = RandomForestClassifier()

    # split the data between training and testing data (90-10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

    # object containing all the possible values for each hyperparameter
    properties = {
        "n_estimators": [x for x in range(20, 100, 20)],
        "max_depth": [x for x in range(20, 120, 20)],
        "min_samples_split": [x for x in range(2, 52, 10)]
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

    cmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cmat,
        display_labels=[str(n) for n in range(10)]
    )

    disp.plot()
    """

    """

    classifier: KNN = KNN()

    # split the data between training and testing data (90-10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=109)

    # object containing all the possible values for each hyperparameter
    properties = {
        "k": [x for x in range(1, 101, 20)],
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
    joblib.dump(tunedParams, '../tuned_models/KNN.pkl')

    # create a new classifier with the best hyperparameters
    bestClassifier = KNN(**tunedParams.best_params_)

    # training the model
    bestClassifier.fit(X_train, y_train)

    X2t, X2ts, y2t, y2ts = train_test_split(X_test, y_test, test_size=0.01, random_state=109)

    # testing the model
    y_pred = bestClassifier.predict(X2ts)

    # print the accuracy
    print("accuracy: ", metrics.accuracy_score(y2ts, y_pred))

    cmat = confusion_matrix(y_true=y2ts, y_pred=y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cmat,
        display_labels=[str(n) for n in range(10)]
    )

    disp.plot()
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=109)

    fit(X_train, y_train)

    X2t, X2ts, y2t, y2ts = train_test_split(X_test, y_test, test_size=0.01, random_state=109)

    y_pred = predict(X2ts)

    # print the accuracy
    print("accuracy: ", metrics.accuracy_score(y2ts, y_pred))

    cmat = confusion_matrix(y_true=y2ts, y_pred=y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cmat,
        display_labels=[str(n) for n in range(10)]
    )

    print(disp)

if __name__ == "__main__":
    main()
