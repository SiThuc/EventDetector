import time
import json
from hubseek.detector import Detector
from hubseek.hubseek import HubSeek
from rank.idfweighter import IDFWeighter
from rank.ranker import Ranker
class BatchDetector(Detector):
    def __init__(self,clustream, graph, config):
        super().__init__(clustream, graph, config)

    def detect(self, td, bandwidth, epsilon, minSup, refSpanTime, eta):
        start = time.time()
        self.init(td, bandwidth, epsilon,eta)
        # use hubseek to get the clusters
        clusters = self.hubseek(minSup)
        print("Hubseek done generating candidates.")
        events = self.rank(clusters, bandwidth, refSpanTime)
        print("There are %d events ranked."%len(events))
        print("Hubseek done ranking events.")

        output = '../output/output_' + str(td.getStartTimestamp())+'_'+str(td.getEndTimestamp())+'.json'
        print(output)
        data = []
        for clus in clusters:
            sub = clus.toJson()
            data.append(sub)

        with open(output, 'w') as f:
            json.dump(data, f)
        f.close()


        for clus in clusters:
            print(clus.__str__())
            print("################################")
        print("Hubseek done generating candidates")

        #rank the cluster as events
        self._events = self.rank(clusters, bandwidth, refSpanTime)
        end = time.time()
        self._timeConsumtion = end - start
        print("Time consumtion in BatchMode is: %d seconds."%self._timeConsumtion)
        self.setStats()

    # init the workers
    def init(self, td, bandwidth, epsilon, eta):
        # self._bTime = td.getStartTimestamp()
        # self._eTime = td.getEndTimestamp()
        self._td = td
        self._hubSeek = HubSeek(bandwidth, epsilon, self._graph)
        weighter = IDFWeighter(td.getTweets())
        self._ranker = Ranker(self._clustream, weighter, eta)
        self._bandwidth = bandwidth
        self._epsilon = epsilon
        self._eta = eta
        self._startTS = td.getStartTimestamp()
        self._endTS = td.getEndTimestamp()

    def setTD(self, tdb):
        self._td = tdb

    # get the candiadate events using hubseek.
    def hubseek(self, minSup):
        tweets = self._td.getTweets()
        self._hubSeek.cluster(tweets)
        return self._hubSeek.genClusters(minSup)

    # rank the clusters with the background knowledge in clustream.
    def rank(self, clusters, bandwidth, refTimeSpan):
        print("Start Timestamp: %d"%self._td.getStartTimestamp())
        print("End Timestamp: %d" % self._td.getEndTimestamp())
        # get the ranking list for the clusters
        scoreCells = self._ranker.rank(clusters, bandwidth, self._td.getStartTimestamp(), self._td.getEndTimestamp(), refTimeSpan)
        # organize the clusters into a ranked order.
        sortedClusters = []
        for sc in scoreCells:
            clusterIndex = sc.getId()
            sortedClusters.append(clusters[clusterIndex])
        return sortedClusters

    #not used for batch detector
    def update(self, deleteTweets, insertTweets, bandwidth, minSup, refTimeSpan, eta, db):
        pass