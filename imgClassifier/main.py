from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from fetchedData import X, y


def main():

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
        classifier, properties, scoring="neg_mean_squared_error", cv=10, return_train_score=True, verbose=4, n_jobs=5
    )

    # start the tuning process
    tunedParams.fit(X_train, y_train)

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


if __name__ == "__main__":
    main()
