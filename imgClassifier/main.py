import os
import time

import joblib
import pandas as pd
import sns
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from KNN import KNN
from fetchedData import X, y


def main():
    # split the data between training and testing data (90-10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=109)

    knnClassfier: KNN = KNN(5, 3)
    knnClassfier.fit(X_train, y_train)

    value = knnClassfier.predict(X_test)

    print("Testing:")
    print('F1 Score: {}'.format(f1_score(y_true=y_test,
                                         y_pred=value,
                                         average='weighted')))


if __name__ == "__main__":
    main()
