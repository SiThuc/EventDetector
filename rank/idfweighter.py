import math
class IDFWeighter(object):
    def __init__(self, tweets):
        self.buildIDF(tweets)

    # build the idfs with all the tweets in the current time window.
    def buildIDF(self, tweets):
        N = len(tweets)
        self._idfs = dict()
        for t in tweets:
            entities = t.getEntities()
            for entity in entities:
                originalCnt = 0
                if entity in self._idfs:
                    originalCnt = self._idfs[entity]
                self._idfs[entity] = originalCnt + 1.0

        for entity in self._idfs.keys():
            n = self._idfs[entity]
            buf = (N - n + 0.5)/(n + 0.5)
            if buf < 0:
                continue
            idf = math.log((N - n + 0.5) / (n + 0.5))
            self._idfs[entity] = idf

    # build the tfs with the tweets in the current cluster.
    def buildTFIDF(self, cluster):
        self._weights = dict() # key: entity Id, value: tf-idf
        # calc tf-idf
        self._tfs = cluster.getEntityOccurrences() #this is tf
        totalweight = 0
        for entity in self._tfs.keys():
            tf = math.log(1.0 + self._tfs[entity])
            tfIdf = tf * self.getIDF(entity)
            self._weights[entity] = tfIdf
            totalweight += tfIdf

        # normalization
        for entity in self._tfs.keys():
            tfIdf = self._weights[entity] / totalweight
            self._weights[entity] = tfIdf
            cluster.setTfIdf(entity, tfIdf)

    def getIDF(self, entity):
        return self._idfs[entity]

    def getWeights(self):
        return self._weights