from hubseek.hubseek import HubSeek
from hubseek.detector import Detector
from geo.tweetdatabase import TweetDatabase
from graph.graph import Graph
from graph.progagator import Propagator
from rank.idfweighter import IDFWeighter
from rank.ranker import Ranker
class OnlineDetector(Detector):
    def __init__(self, graph, config):
        super().__init__(graph, config)

    def update(self, deleteTD, insertTD, bandwidth, minSup, refTimeSpan, eta):
        clusters = self.updateClusters(deleteTD, insertTD, minSup, refTimeSpan, eta)
        self.updateTweetDatabase(deleteTD, insertTD)
        self.updateRanker(eta)
        # rank the clusters as events
        events = self.rank(clusters, bandwidth, refTimeSpan)
        return self._td

    def updateClusters(self, deleteTd, insertTd, minSup, db):
        self._hubSeek.delete(deleteTd.getTweets())
        self._hubSeek.insert(insertTd.getTweet)
        return self._hubSeek.genClusters(minSup)

    def updateTweetDatabase(self, deleteTd, insertTd):
        self._td.deleteFromHead(deleteTd.size())
        self._addAll(insertTd)

    def updateRanker(self, eta):
        weighter = IDFWeighter(self._td.getTweets())
        ranker = Ranker(self._clustream, weighter, eta)



























