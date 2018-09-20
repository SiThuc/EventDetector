import time
from hubseek.tweetcluster import TweetCluster
class HubSeek(object):
    def __init__(self, bandwidth, epsilon, entityGraph):
        self._bandwidth = bandwidth
        self._epsilon = epsilon
        self._entityGraph = entityGraph

    def setEntityGraph(self, newGraph):
        self._entityGraph = newGraph

    # cluster the given points in a batch mode
    def cluster(self, inputData):
        start = time.time()
        self.init(inputData)
        self.calcNeighborhood()
        self.calcScores()
        self.findLocalHubs()

        end = time.time()
        self._numBatchTweets = len(inputData)
        self._timeBatchClustering  = end-start

    def init(self, inputData):
        self._points = dict()
        for t in inputData:
            self._points[t.getTweetId()] = t

        #init neighborhood
        self._neighborhood = dict()
        #init the rest arrays
        self._scores = dict()
        self._localHubs = dict()
        self._globalHubs = dict()

    def calcNeighborhood(self):
        for tid in self._points.keys():
            #find the in-neighbots among points for e
            inNeighbors = self.findInNeighbors(tid)
            self._neighborhood[tid] = inNeighbors

    def calcScores(self):
        for tid in self._points.keys():
            # calc score for e using the current points
            score = self.calcScore(tid)
            self._scores[tid] = score

    def findLocalHubs(self):
        for tid in self._points.keys():
            localHub = self.findLocalHubForOnePoint(tid)
            self._localHubs[tid] = localHub

    # delete some points, update: points, neighborhood, and score
    def delete(self, oldPoints):
        start = time.time()
        self.initForDelete(oldPoints)
        # get the out neighborhood for the points that need to be deleted
        outNeighborhood = self.findOutNeighborhood(oldPoints)
        # update the in-neighborhood for the existing points
        self.updateNeighborhoodForDelete(outNeighborhood)
        # update the scores for existing points, note that the out neighborhood is defined on updated points
        self.updateScoresForDelete(oldPoints, outNeighborhood)
        # update the local hubs.
        self.updateLocalHubs()
        # write the stats
        end = time.time()
        self._numDeletedTweet = len(oldPoints)
        self._timeDeletion = end - start

    #     // init the points and other data fields
    #     // parameter: the old points that need to be deleted.
    def initForDelete(self, oldPoints):
        for e in oldPoints:
            tid = e.getTweetId()
            del self._points[tid]
            del self._neighborhood[tid]
            del self._scores[tid]
            del self._localHubs[tid]
            del self._globalHubs[tid]

    # update the neighborhood. Input: the out neighborhood for the old points.
    def updateNeighborhoodForDelete(self, outNeighborhood):
        for tid in outNeighborhood.keys():
            outNeighbors = outNeighborhood[tid]
            for neighbor in outNeighbors:
                # remove e from the in-neighborhood
                #del self._neighborhood[neighbor][tid]
                self._neighborhood[neighbor].remove(tid)


    # return the remaining tweets whose scores have been updated
    def updateScoresForDelete(self, oldPoints, outNeighborhood):
        for e in oldPoints:
            # update the scores for the out neighbors of e.
            outNeighborIds = outNeighborhood.get(e.getTweetId())
            for neighborId in outNeighborIds:
                neighbor = self._points[neighborId]
                deltaScore = neighbor.calcScoreFrom(e, self._bandwidth, self._entityGraph)
                updatedScore = self._scores[neighborId] - deltaScore
                self._scores[neighborId] = updatedScore

    # update the local hubs for the remaining points
    def updateLocalHubs(self):
        for e in self._points.keys():
            localHub = self.findLocalHubForOnePoint(e)
            self._localHubs[e] = localHub

    # insert new data
    def insert(self, insertData):
        start = time.time()
        # update the points
        self.initForInsert(insertData)
        # get the out neighborhood for the points that need to be inserted
        outNeighborhood = self.findOutNeighborhood(insertData)
        # update the in-neighborhood for the existing points
        self.updateNeighborhoodInsert(outNeighborhood)
        # update the scores for existing points
        self.updateScoresInsert(insertData, outNeighborhood)
        # update the local hubs.
        self.updateLocalHubs()
        # write the stats
        end = time.time()
        self._numInsertedTweet = len(insertData)
        self._timeInsertion = end - start

    # update the fields for insertion.
    def initForInsert(self, insertData):
        for e in insertData:
            tid = e.getTweetId()
            self._points[tid] = e
            self._neighborhood[tid] = set()
            self._scores[tid] = 0
            self._localHubs[tid] = -1
            self._globalHubs[tid] = -1

    def updateNeighborhoodInsert(self, outNeighborhood):
        for e in outNeighborhood.keys():
            # add e into the in-neighbor set for e's out-neighbors
            outNeighbors = outNeighborhood[e]
            for neighbor in outNeighbors:
                self._neighborhood[neighbor].add(e)
            # create the in-neighbor set for e itself
            inNeighbors = self.findInNeighbors(e)
            self._neighborhood[e] = inNeighbors

    # return the set of geo tweets whose scores have been updated
    def updateScoresInsert(self, insertData, outNeighborhood):
    # Update the scores for the old points
        for e in insertData:
        # update the scores for the out neighbors of e.
            outNeighborIds = outNeighborhood[e.getTweetId()]
            for neighborId in outNeighborIds:
                neighbor = self._points[neighborId]
                deltaScore = neighbor.calcScoreFrom(e, self._bandwidth, self._entityGraph)
                #if the neighbor is the entity itself and the entity is new, then old score is 0.
                updatedScore = self._scores[neighborId] + deltaScore
                self._scores[neighborId] = updatedScore
        # calc the scores for the new points
        for e in insertData:
            #note that: the new points in the neighborhood is recomputed
            score = self.calcScore(e.getTweetId())
            self._scores[e.getTweetId()] = score

    # generate the clusters
    def genClusters(self, supportThreshold):
        self.findGlobalHubs()
        clusters = dict() # key: mode, value: cluster
        for e in self._points.values():
            globalHub = self._globalHubs[e.getTweetId()]
            if globalHub in clusters:
                gec = clusters[globalHub]
                gec.add(e, self._scores[e.getTweetId()])
            else:
                # create a new cluster centered at the mode
                hub = self._points[globalHub]
                gec = TweetCluster(hub)
                gec.add(e, self._scores[e.getTweetId()])
                clusters[globalHub] = gec

        # prune the clusters by size.
        results = []
        for k,v in clusters.items():
            c = v
            if c.size() >= supportThreshold:
                results.append(c)
        return results

    def findGlobalHubs(self):
        for e in self._points.keys():
            cnt = 0
            currentPoint = e
            while True:
                localHub = self._localHubs[currentPoint]
                if currentPoint == localHub:
                    break
                cnt += 1
                if cnt >= 1000:
                    print("current:" + str(currentPoint) + "local hub:" + str(localHub))
                    print(self._scores[currentPoint])
                    print(self._neighborhood[currentPoint])
                    print(self._neighborhood[localHub])
                if cnt >= 1010:
                    print("Finding global hubs error.")
                    exit(1)
                currentPoint = localHub
            self._globalHubs[e] = currentPoint
    # utils functions
    # find the in-neighbors for one geo-tweet
    def findInNeighbors(self, tid):
        neighbors = set()
        e = self._points[tid] # query tweet
        for other in self._points.values():
            geoDist = e.calcGeoDist(other)
            graphProximity = e.calcGraphDistFrom(self._entityGraph, other)
            if geoDist <= self._bandwidth and graphProximity >= self._epsilon:
                neighbors.add(other.getTweetId())
        #neighbors.add(tid); // add the tweet itself into the set
        if len(neighbors) == 0:
            print("no neighbor:" + str(e.getEntities()))
        return neighbors

    # find the out-neighbors for one geo-entity
    def findOutNeighbors(self, e):
        neighbors = set()
        for other in self._points.values():
            geoDist = e.calcGeoDist(other)
            graphProximity = other.calcGraphDistFrom(self._entityGraph, e)
            if geoDist <= self._bandwidth and graphProximity >= self._epsilon:
                neighbors.add(other.getTweetId())
        return neighbors

    def findOutNeighborhood(self, insertTweets):
        outNeighborhood = dict()
        for e in insertTweets:
            outNeighbors = self.findOutNeighbors(e)
            outNeighborhood[e.getTweetId()] = outNeighbors
        return outNeighborhood

    def calcScore(self, tid):
        score = 0
        e = self._points[tid]
        neighborIds = self._neighborhood[tid]
        for nid in neighborIds:
            neighbor = self._points[nid]
            score += e.calcScoreFrom(neighbor, self._bandwidth, self._entityGraph)
        return score

    def findLocalHubForOnePoint(self, tid):
        localHub = None
        maxScore = -1.0
        neighbors = self._neighborhood[tid]
        # if the tweet does not have any neighbor at all, return the tweet itself.
        if len(neighbors) == 0:
            return tid
        for neighbor in neighbors:
            score = self._scores[neighbor]
            if score > maxScore:
                maxScore = score
                localHub = neighbor
            elif score == maxScore and not(localHub is None) and neighbor > localHub:
                localHub = neighbor
        return localHub

    def printStats(self):
        s = "HubSeek Stats:"
        s += " numBatchTweets=" + str(self._numBatchTweets)
        #s += "; numDeletedTweets=" + str(self._numDeletedTweet)
        #s += "; numInsertedTweets=" + str(self._numInsertedTweet)
        s += "; timeBatchClustering=" + str(self._timeBatchClustering)
        #s += "; timeDeletion=" + str(self._timeDeletion)
        #s += "; timeInsertion=" + str(self._timeInsertion)
        print(s)

    def getNumBatchTweets(self):
        return self._numBatchTweets

    def getNumDeletedTweet(self):
        return self._numDeletedTweet

    def getNumInsertedTweet(self):
        return self._numInsertedTweet

    def getTimeBatchClustering(self):
        return self._timeBatchClustering

    def getTimeDeletion(self):
        return self._timeDeletion

    def getTimeInsertion(self):
        return self._timeInsertion