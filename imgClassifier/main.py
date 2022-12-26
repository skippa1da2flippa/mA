import os
import time

import joblib
import pandas as pd
import sns
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from KNN import KNN
from fetchedData import X, y, X_train, y_train


def main():
    print("inzio main")
    knnClassfier: KNN = KNN(5, 3)
    knnClassfier.fit(X_train, y_train)

    print("finito allenamento")

    print("inizio test")
    value = knnClassfier.predict(X_train)

    print("Testing:")
    print(''.format(f1_score(y_true=y_train,
                                         y_pred=value,
                                         average='weighted')))
    print("finsco main")


if __name__ == "__main__":
    main()
