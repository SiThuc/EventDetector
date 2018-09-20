
class ReferencePeriod(object):

    def __init__(self, startSnapshot, endSnapshot):
        self._startSnapshot = startSnapshot
        self._endSnapshot = endSnapshot
        self._startTimestamp = startSnapshot.getTimestamp()
        self._endTimestamp = endSnapshot.getTimestamp()
        self._clusters = list(endSnapshot.getDiffClusters(startSnapshot))
        self._smoothingCnt = 0.001
        self._weights = list()

    # get the expected number of occurrences for a given set of keywords at a location during a time span.
    def getExpectedOccurrences(self, words, loc, bandwidth,timeSpan):
        self.calcWeights(loc, bandwidth)
        scalingFactor = self.getScalingFactor(timeSpan)
        expectedOccurrences = dict()
        for word in words:
            interpolation = self.getEstimatedNumber(word)
            expectedOccurrences[word] = interpolation * scalingFactor
        return expectedOccurrences

    # calculate the weights of the cluster centers to the query location
    def calcWeights(self, loc, bandwidth):
        self._weights = list()
        for cluster in self._clusters:
            dist = cluster.getCentroid().getDistance(loc.toRealVector())
            weight = 0
            if dist < bandwidth:
                weight = 1.0 - (dist / bandwidth) * (dist / bandwidth)
            self._weights.append(weight)

    def getScalingFactor(self, timeSpan):
        return float(timeSpan / float(self._endTimestamp - self._startTimestamp))

    # use kernel to get the estimated occurrence for the given word
    def getEstimatedNumber(self, word):
        result = 0
        for i in range(0,len(self._clusters)):
            cluster = self._clusters[i]
            weight = self._weights[i]
            wordsInCluster = cluster.getWords()
            wordCntInCluster = self._smoothingCnt
            if word in wordsInCluster:
                wordCntInCluster = wordsInCluster[word]
            result += weight * wordCntInCluster
        return result