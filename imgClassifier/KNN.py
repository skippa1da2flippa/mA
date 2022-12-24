from math import sqrt

nullVector: list[float] = [0.0 for i in range(0, 784)]


def euclideanDistance(pointA: list[float], pointB: list[float]):
    distance: float = 0.0
    for i in range(0, 784):
        distance += (pointA[i] - pointB[i]) ** 2

    return sqrt(distance)


class PointNdDistance:
    def __init__(self, point: list[float], distance: float):
        self.point = point
        self.distance = distance


def wrapper(comparingPoint: list[float], fromTheOrigin: bool = False):
    def comparingWEdge(point1: PointNdDistance, point2: PointNdDistance):
        point1Distance: float = 0.0
        point2Distance: float = 0.0

        if not fromTheOrigin:
            point1Distance = euclideanDistance(comparingPoint, point1.point)
            point2Distance = euclideanDistance(comparingPoint, point2.point)
        else:
            point1Distance = point1.distance
            point2Distance = point2.distance

        if point1Distance > point2Distance:
            return 1
        elif point1Distance == point2Distance:
            return 0
        else:
            return -1

    return comparingWEdge


class Neighborhood:
    # shape tells you how much margin are we taking for grabbing the right contour(the lower,the fewer points are taken)
    shape: float = 0.7

    # edge accuracy tells you how much accuracy we want to create the contour (the lower, the more precise)
    edgeAccuracy: float = 0.3

    def __init__(self, label: str):
        self.label = label
        self.points: list[PointNdDistance] = []
        self.furthestPoint = []
        self.closestPoint = []
        self.furthestPointDistance: float = 0.0
        self.closestPointDistance: float = 0.0

    def addPoint(self, point: list[float]):
        newDistance = euclideanDistance(nullVector, point)
        if newDistance < self.closestPointDistance:
            self.closestPoint = point
            self.closestPointDistance = newDistance

        if newDistance > self.furthestPointDistance:
            self.furthestPoint = point
            self.furthestPointDistance = newDistance

        self.points.append(PointNdDistance(point, newDistance))

    def __pointSorter(self):
        self.points.sort(key=wrapper(self.closestPoint))

    def findLeftNdRightBound(self, point: list[float]) -> tuple[int, int]:
        startingLine: int = -1
        endLine: int = -1
        lowerDistance: float = euclideanDistance(self.closestPoint, point)
        upperDistance: float = euclideanDistance(self.furthestPoint, point)
        originDistance: float = euclideanDistance(nullVector, point)

        if lowerDistance > upperDistance:
            if upperDistance / lowerDistance > Neighborhood.edgeAccuracy:
                startingLine = self.findBestStart(originDistance, False)
                endLine = self.findBestEnd(originDistance, startingLine, False)
            else:
                startingLine = len(self.points) - 1
                endLine = int(len(self.points) - (2 * (len(self.points) * Neighborhood.edgeAccuracy)))
        else:
            if lowerDistance / upperDistance > Neighborhood.edgeAccuracy:
                startingLine = self.findBestStart(originDistance)
                endLine = self.findBestEnd(originDistance, startingLine)
            else:
                startingLine = 0
                endLine = int(startingLine + (2 * (len(self.points) * Neighborhood.edgeAccuracy)))

        return startingLine, endLine

    def findBestStart(self, distance: float, fromTheFront: bool = True):
        if fromTheFront:
            myRange = range(0, len(self.points))
            for i in myRange:
                if self.points[i].distance / distance >= Neighborhood.shape:
                    return i
        else:
            myRange = range(len(self.points) - 1, 0, step=-1)
            for i in myRange:
                if distance / self.points[i].distance >= Neighborhood.shape:
                    return i

    def findBestEnd(self, distance: float, startingIdx: int, fromTheFront: bool = True):
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

    def getTheCandidates(self, point: list[float], k: int):
        leftRightEnd: tuple[int, int] = self.findLeftNdRightBound(point)
        candidates: list[PointNdDistance] = []
        for i in range(leftRightEnd[0], leftRightEnd[1] + 1):
            candidates.append(self.points[i])

        candidates.sort(key=wrapper(point))

        return [candidates[i].point for i in range(len(candidates) - 1, len(candidates) - (k + 1))]
