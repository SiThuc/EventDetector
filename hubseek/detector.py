from abc import ABC

class Detector(ABC):


    def __init__(self,clustream, graph, config):
        self._clustream = clustream
        self._graph = graph
        self._config = config
        self._hubSeek = NotImplemented
        self._td = NotImplemented
        self._ranker = NotImplemented

    def setStats(self):
        #self._numEvents = len(self.cluster)
        #self._numTweetsInClustream = self._clustream.getNumOfProcessedTweets()
        self._numTweetsHubSeek = self._hubSeek.getNumBatchTweets()
        #self._numTweetsHubSeekDeletion = self._hubSeek.getNumDeletedTweet()
        #self._numTweetsHubSeekInsertion = self._hubSeek.getNumInsertedTweet()
        #self._numReferencePeriods = self._ranker.getNumReferencePeriods()
        #self._timeClustream = self._clustream.getElapsedTime()
        self._timeGraphVicinity = self._graph.getTimeCalcVicinity()
        self._timeHubSeekBatch = self._hubSeek.getTimeBatchClustering()
        #self._timeHubSeekDeletion = self._hubSeek.getTimeDeletion()
        #self._timeHubSeekInsertion = self._hubSeek.getTimeInsertion()

    def detect(self, td, bandwidth, epsilon, munSup, eta):
        raise NotImplementedError("Subclass must implement abstract method")

    def update(self, deleteTweets, insertTweets, bandwidth, munSup, refTimeSpan, eta, db):
        raise NotImplementedError("Subclass must implement abstract method")

    def setClustream(self, clustream):
        self._clustream = clustream

    def printStats(self):
        self._clustream.printStats()
        self._hubSeek.printStats()
        #self._ranker.printStats()




