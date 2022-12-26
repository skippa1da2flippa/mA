from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, Protocol, Any, Mapping, List, NamedTuple, Dict, Callable

import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import sqrt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from imgClassifier.KNN import PointNdDistance, FullPoint


# comparing algorithm between to fullPoint
def comparingFullPoint(fstElem: FullPoint):
    return fstElem.distance


# this function compute the euclidean distance between two points
def euclideanDistance(pointA: list[float], pointB: list[float]) -> float:
    distance: float = 0.0
    for i in range(0, 784):
        distance += (pointA[i] - pointB[i]) ** 2

    return sqrt(distance)


# comparing algorithm between to PointNdDistance
def comparingPointNdDistance(fstElem: PointNdDistance):
    return fstElem.distance


class BaseClassifier(BaseEstimator, ClassifierMixin, ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> BaseClassifier:
        pass

    @abstractmethod
    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        pass


class KNN(BaseClassifier):
    X_: np.ndarray = None
    """
    Training samples
    """

    y_: np.ndarray = None
    """
    Training labels
    """

    labels_: np.ndarray = None
    """
    Unique labels
    """

    k: int
    """
    Number of nearest neighbors to consider
    """

    def __init__(self, k: int = 1):
        self.k: int = k
        self.lazyList: list[FullPoint] = []

    def fit(self, X, y) -> KNN:
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        for i in range(X.shape[0]):
            self.lazyList.append(FullPoint(X[i, :], -1, self.y_[i]))

        # Store the classes seen during fit
        self.labels_ = unique_labels(y)

        # Return the classifier
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        return (
                self.X_ is not None
                and self.y_ is not None
                and self.labels_ is not None
        )

    def predict(self, X) -> np.ndarray:
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation: array should be 2D with shape (n_samples, n_features)
        check_array(X)

        predictions = []

        print(type(X))

        X = X.to_numpy()

        for i in range(X.shape[0]):
            # st.mode with the below parameters returns a named tuple with fields ("mode", "count"),
            #   each of which has a single value (because keepdims = False)
            prediction: Tuple[np.ndarray, np.ndarray] = st.mode(a=self.findBestK(X[i, :]),
                                                                axis=None, keepdims=False)
            print("found one")
            predictions.append(prediction.mode)

        return np.array(predictions)

    def findBestK(self, point: list[int]):
        for j in range(len(self.lazyList)):
            self.lazyList[j].distance = euclideanDistance(point, self.lazyList[j].point)

        self.lazyList.sort(key=comparingFullPoint)
        occurrences: list[int] = [0 for i in range(0, 10)]

        for k in range(0, self.k):
            occurrences[self.lazyList[k].label] += 1

        maximum: int = 0
        idj: int = -1

        for w in range(0, 10):
            if occurrences[w] > maximum:
                maximum = occurrences[w]
                idj = w

        return idj
