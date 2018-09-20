import random
from geo.ArrayRealVector import ArrayRealVector
class Kmeans(object):

    def __init__(self, maxIter):
        self._prevMean = None
        self._mean = None
        self._clusters = None
        self._data = None
        self._weights = None

        self._maxIter = maxIter
#        self._r = random.seed(100)

    def cluster(self, data, weights, K):
        self.initialize(data, weights, K)
        for iteration in range(0, self._maxIter):
            self.computeMean()
            if not self.hasMeanChanged():
                break
            self.assignToClusters()
        return self._clusters



    def initialize(self, data, weights, K):
        # number of clusters
        if len(data) < K :
            print("Warning: fewer data points than number of clusters.")
            self._nCluster = len(data)
        else:
            self._nCluster = K
        #data
        self._data = data
        #weights
        if weights != None:
            self._weights = weights
        else:
            # equal weights
            self._weights = []
            for i in range(0, len(data)):
                self._weights.append(1.0)
        # intermediate arrays
        self._prevMean = []
        self._mean = self.getRandomMean()
        # results
        self._clusters = []
        for i in range(0, K):
            self._clusters.insert(i,[])
        self.assignToClusters()

    def getRandomMean(self):
        randomPoints = set()
        n = len(self._data)
        completeArray = []
        for i in range(0, n):
            completeArray.insert(i,i)
        bound = n
        while len(randomPoints) < self._nCluster:
            randNum = random.randint(0, bound - 1)
            randomPoints.add(self._data[randNum])
            completeArray[randNum] = completeArray[bound - 1]
            bound -= 1
        return list(randomPoints)

    def assignToClusters(self):
        for i in range(0, len(self._clusters)):
            self._clusters[i] = list()
        for i in range(0, len(self._data)):
            assignId = self.getNearestCluster(self._data[i])
            self._clusters[assignId].append(i)

    def getNearestCluster(self, p):
        result = 0
        minDist = self._mean[0].getDistance(p)
        for i in range(1, self._nCluster):
            dist = self._mean[i].getDistance(p)
            if dist <= minDist:
                minDist = dist
                result = i
        return result

    def computeMean(self):
        # before computing, store current version of mean into previous mean
        for i in range(0, self._nCluster):
            if len(self._prevMean) < i:
                self._prevMean[i] = self._mean[i].copy()
            else:
                self._prevMean.append(self._mean[i].copy())

        for i in range(0, self._nCluster):
            sumWeight = 0.0
            self._mean[i] = ArrayRealVector(self._prevMean[i].getDimension())
            dataIds = self._clusters[i]
            for dataId in dataIds:
                weight = self._weights[dataId]
                self._mean[i] = self._mean[i].add(self._data[dataId].mapMultiply(weight))
                sumWeight += weight
            self._mean[i].mapDivideToSelf(sumWeight)


    def hasMeanChanged(self):
        for i in range(0, self._nCluster):
            if self._prevMean[i].__eq__(self._mean[i]):
                return True
        return False






















