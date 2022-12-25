from abc import ABC, abstractmethod
from math import sqrt
from typing import Tuple
import scipy.stats as st
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

# null vector used as origin also called o or w here below
nullVector: list[float] = [0.0 for i in range(0, 784)]


# this function compute the euclidean distance between two points
def euclideanDistance(pointA: list[float], pointB: list[float]) -> float:
    distance: float = 0.0
    for i in range(0, 784):
        distance += (pointA[i] - pointB[i]) ** 2

    return sqrt(distance)


# class used to represent a point and its related label
class PointNdLabel:
    def __int__(self, point: list[float], label: int):
        self.point = point
        self.label = label


# class used to wrap a point and its distance to a specific point
class PointNdDistance:
    def __init__(self, point: list[float], distance: float):
        self.point = point
        self.distance = distance


# class used to wrap a point, its label and its distance to a specific point
class FullPoint:
    def __int__(self, point: list[float], distance: float, label: int):
        self.point = point
        self.distance = distance
        self.label = label


# comparing algorithm between to tuple
def comparingNeighborhood(fstElem: tuple[PointNdDistance, int], sdnElem: tuple[PointNdDistance, int]):
    if fstElem[0].distance > sdnElem[0].distance:
        return 1
    elif fstElem[0].distance == sdnElem[0].distance:
        return 0
    else:
        return -1


# comparing algorithm between to fullPoint
def comparingFullPoint(fstElem: FullPoint, sdnElem: FullPoint):
    if fstElem.distance > sdnElem.distance:
        return 1
    elif fstElem.distance == sdnElem.distance:
        return 0
    else:
        return -1


# this function return a comparing algorithm for points
def wrapper(comparingPoint: list[float], fromTheOrigin: bool = False, saveDistance: bool = False):
    def comparingWEdge(point1: PointNdDistance, point2: PointNdDistance):
        point1Distance: float = 0.0
        point2Distance: float = 0.0

        if not fromTheOrigin:
            point1Distance = euclideanDistance(comparingPoint, point1.point)
            point2Distance = euclideanDistance(comparingPoint, point2.point)
        else:
            point1Distance = point1.distance
            point2Distance = point2.distance

        if saveDistance:
            point1.distance = point1Distance
            point2.distance = point2Distance

        if point1Distance > point2Distance:
            return 1
        elif point1Distance == point2Distance:
            return 0
        else:
            return -1

    return comparingWEdge


# class used to represent a neighborhood inside a multidimensional space
class Neighborhood:
    # shape tells you how much margin are we taking for grabbing
    # the right contour (the higher,the fewer points are taken)
    shape: float = 0.7

    # edge accuracy tells you how much accuracy we want to create the contour (the lower, the more precise)
    edgeAccuracy: float = 0.3

    def __init__(self, label: int):
        self.label = label
        self.points: list[PointNdDistance] = []
        self.furthestPoint = []
        self.closestPoint = []
        self.furthestPointDistance: float = 0.0
        self.closestPointDistance: float = 0.0

    # this method adds a point to the neighborhood points list it also look for the closes and the furthest point from w
    def addPoint(self, point: list[float]):
        newDistance = euclideanDistance(nullVector, point)
        if newDistance < self.closestPointDistance:
            self.closestPoint = point
            self.closestPointDistance = newDistance

        if newDistance > self.furthestPointDistance:
            self.furthestPoint = point
            self.furthestPointDistance = newDistance

        for i in range(0, len(self.points)):
            if self.points[i].distance > newDistance:
                self.points.insert(i, PointNdDistance(point, newDistance))

    # this method find the right slice of the points list to use as nearest points to the input point
    def __findLeftNdRightBound(self, point: list[float]) -> tuple[int, int]:
        startingLine: int = -1
        endLine: int = -1
        lowerDistance: float = euclideanDistance(self.closestPoint, point)
        upperDistance: float = euclideanDistance(self.furthestPoint, point)
        originDistance: float = euclideanDistance(nullVector, point)

        if lowerDistance > upperDistance:
            # if the shorter distance is not even a specified percentage of the longer one
            if upperDistance / lowerDistance > Neighborhood.edgeAccuracy:
                startingLine = self.__findBestStart(originDistance, False)
                endLine = self.__findBestEnd(originDistance, startingLine, False)
            else:
                # then the right margin is the furthest point from o
                startingLine = len(self.points) - 1
                # the left margin is given by two times the len of the (len*edgeAccuracy)
                endLine = int(len(self.points) - (2 * (len(self.points) * Neighborhood.edgeAccuracy)))
        else:
            # same thing here
            if lowerDistance / upperDistance > Neighborhood.edgeAccuracy:
                startingLine = self.__findBestStart(originDistance)
                endLine = self.__findBestEnd(originDistance, startingLine)
            else:
                startingLine = 0
                endLine = int(startingLine + (2 * (len(self.points) * Neighborhood.edgeAccuracy)))

        return startingLine, endLine

    # this method find the first index to use as right/left slice
    def __findBestStart(self, distance: float, fromTheFront: bool = True) -> int:
        if fromTheFront:
            myRange = range(0, len(self.points))
            for i in myRange:
                # getting the first point whose distance is at least a specific hyperparameter "shape"
                if self.points[i].distance / distance >= Neighborhood.shape:
                    return i
        else:
            myRange = range(len(self.points) - 1, 0, step=-1)
            for i in myRange:
                if distance / self.points[i].distance >= Neighborhood.shape:
                    return i

    # this method find the last index to use as right/left slice
    def __findBestEnd(self, distance: float, startingIdx: int, fromTheFront: bool = True) -> int:
        if fromTheFront:
            myRange = range(startingIdx, len(self.points))
            for i in myRange:
                if distance / self.points[i].distance <= Neighborhood.shape:
                    return i
        else:
            myRange = range(startingIdx, 0, step=-1)
            for i in myRange:
                if self.points[i].distance / distance <= Neighborhood.shape:
                    return i

    # this method find the k-nearest candidates to the input point
    def getTheCandidates(self, point: list[float], k: int) -> list[PointNdDistance]:
        leftRightEnd: tuple[int, int] = self.__findLeftNdRightBound(point)
        candidates: list[PointNdDistance] = []

        if leftRightEnd[0] < leftRightEnd[1]:
            leftEnd = leftRightEnd[0]
            rightEnd = leftRightEnd[1]
        else:
            rightEnd = leftRightEnd[0]
            leftEnd = leftRightEnd[1]

        for i in range(leftEnd, rightEnd + 1):
            candidates.append(self.points[i])

        candidates.sort(key=wrapper(point, saveDistance=True))

        return [candidates[i] for i in range(0, k)]


class BaseClassifier(BaseEstimator, ClassifierMixin, ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

    @abstractmethod
    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        pass


class KNN(BaseClassifier, ABC):

    def __init__(self, k: int, neighborhoodAvoidance: int, shape: float = 0.7, edgeAccuracy: float = 0.3):
        self.labels_ = None
        self.y_ = None
        self.X_ = None
        self.k = k
        self.neighborhoods: list[Neighborhood] = []
        self.neighborhoodAvoidance = neighborhoodAvoidance
        self.kPoints: list[PointNdLabel] = []

        for i in range(0, 10):
            self.neighborhoods.append(Neighborhood(i))

        Neighborhood.edgeAccuracy = edgeAccuracy
        Neighborhood.shape = shape

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Store the classes seen during fit
        self.labels_ = unique_labels(y)

        # fill the 9 different neighborhoods
        for idx, x in np.ndenumerate(self.X_):
            self.neighborhoods[self.y_[idx]].addPoint(x.toList())

        # Return the classifier
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        return (
                self.X_ is not None
                and self.y_ is not None
                and self.labels_ is not None
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation: array should be 2D with shape (n_samples, n_features)
        check_array(X)

        predictions = []

        with np.nditer(X, op_flags=['readwrite']) as it:
            for x in it:
                # st.mode with the below parameters returns a named tuple with fields ("mode", "count"),
                #   each of which has a single value (because keepdims = False)
                prediction: Tuple[np.ndarray, np.ndarray] = st.mode(a=self.findClosestNeighborhoods(x.toList()),
                                                                    axis=None, keepdims=False)
                predictions.append(prediction.mode)

        return np.array(predictions)

    # find the closest neighborhoods to the input point
    def findClosestNeighborhoods(self, point: list[float]):
        distances: list[tuple[PointNdDistance, int]] = []
        candidates: list[tuple[list[PointNdDistance], int]] = []

        for neighbor in self.neighborhoods:
            furthestPointDist: float = euclideanDistance(neighbor.furthestPoint, point)
            closestPointDist: float = euclideanDistance(neighbor.closestPoint, point)

            if furthestPointDist > closestPointDist:
                distances.append((PointNdDistance(point, closestPointDist), neighbor.label))
            else:
                distances.append((PointNdDistance(point, furthestPointDist), neighbor.label))

        distances.sort(key=comparingNeighborhood)

        # neighborhoodAvoidance store the maximum number of close neighborhoods taken as feasible ones
        for i in range(0, self.neighborhoodAvoidance):
            candidates.append((self.neighborhoods[distances[i][1]].getTheCandidates(point, self.k), distances[i][1]))

        return self.prediction(candidates)

    # predict the label of the input point by sorting all the best candidates from the different neighborhoods
    # and use the k-best
    def prediction(self, candidates: list[tuple[list[PointNdDistance], int]]):
        tempList: list[FullPoint] = []
        for i in range(0, len(candidates)):
            for j in range(0, len(candidates[i][0])):
                fullPoint = FullPoint(candidates[i][0][j].point, candidates[i][0][j].distance, candidates[i][1])
                tempList.append(fullPoint)

        occurrences: list[int] = [0 for i in range(0, 8)]

        tempList.sort(key=comparingFullPoint)

        for i in range(0, self.k):
            occurrences[tempList[i].label] += 1

        maxTup: tuple[int, int] = (-1, -1)
        maximum = occurrences[0]

        # majority vote
        for j in range(0, 10):
            if occurrences[j] > maximum:
                maximum = occurrences[j]
                maxTup = (maximum, j)

        return maxTup[1]
