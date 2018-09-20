import numpy as np
import sys
import time
import math
import heapq
from utils.utils import Utils
from clustream.pyramid import Pyramid
from clustream.kmeans import Kmeans
from clustream.microcluster import MicroCluster
from clustream.microclusterpair import MicroClusterPair
from geo.ArrayRealVector import ArrayRealVector

class Clustream(object):
    # maxNumCluster: max number of clusters that trigger merge operation, and the merge coefficient
    # numTweetPeriod: number of tweets, used to periodically delete outdated clusters.
    # outdatedThreshold: to test whether a cluster is outdated.
    def __init__(self, maxNumCluster = 200, numTweetPeriod = 2000, outdatedThreshold = 432000):
        # to test whether a tweet should form a new cluster
        self._mbsFactor = 0.8
        self._mc = 0.7
        # number of tweets that have been processed so far.
        self._tweetCnt = 0
        self._quantile = Utils.getQuantile(0.95)
        # current timestamp
        self._currentTimestamp = -1
        # cluster id that is to be assigned
        self._toAssignClusterId = 0
        # the clusters
        self._clusters = dict()
        self._ptf = Pyramid()
        # stats
        self._numOfProcessedTweets = 0
        self._noHitCount = 0
        self._mergeNumber = 0
        self._elapsedTime = 0
        self._outdatedCount = 0

        self._maxNumCluster = maxNumCluster
        self._numTweetPeriod = numTweetPeriod
        self._outdatedThreshold = outdatedThreshold
    # ***************************  Initialization ****************************
    def init(self, initialTweets, initNumCluster):
        # kmean clustering
        kmeansResults = self.kmeansClustering(initialTweets, initNumCluster)
        # generate initial geoTweet clusters
        self.genInitialGeoTweetClusters(initialTweets, kmeansResults)

    def kmeansClustering(self, initialTweets, initNumCluster):
        kmeansMaxIter = 50
        kmeans = Kmeans(kmeansMaxIter)
        # convert the tweets to real vectors to perform kmeans clustering.
        data = []
        for tweet in initialTweets:
            data.append(tweet.getLocation().toRealVector())
        # weights are null
        return kmeans.cluster(data, None, initNumCluster)

    def genInitialGeoTweetClusters(self, tweets, clusteringResults):
        for memberTweetIndices in clusteringResults:
            memberTweets = []
            for index in memberTweetIndices:
                memberTweets.append(tweets[index])
            cluster = MicroCluster.MicroCluster_from_tweets_id(memberTweets, self._toAssignClusterId)
            self._clusters[self._toAssignClusterId] = cluster
            self._toAssignClusterId += 1

    #*****************************Clustering**********************************
    # Note: the tweets should come in the ascending order of timestamp.
    def update(self, tweet):
        start = time.time()
        self._currentTimestamp = tweet.getTimestamp()
        self._tweetCnt += 1
        # 2.1 try to absorb this tweet to existing clusters
        chosenID = self.findToMergeCluster(tweet.getLocation().toRealVector())
        if chosenID < 0:        #2.1.1 no close cluster exists, create a new cluster
            self.createNewCluster(tweet)
        else:                   #2.1.2 Data fits, put into cluster and be happy
            self._clusters[chosenID].absorb(tweet)
        # 2.2 periodically check outdated clusters
        if (self._tweetCnt % self._numTweetPeriod) == 0:
            self.removeOutdated()

        #2.3 if cluster number reach a limit, merge clusters
        if(len(self._clusters) > self._maxNumCluster):
            self.mergeCluster()
            self._mergeNumber += 1
        # 2.4 update the pyramid structure if necessary.
        self.updatePyramid()
        # 2.5 update the stats
        end = time.time()
        self._elapsedTime += (end - start)
        self._numOfProcessedTweets += 1
        if self._numOfProcessedTweets % 10000 == 0:
            print("Clustream time for %d Tweets is %d"%(self._numOfProcessedTweets, self._elapsedTime))

    def findToMergeCluster(self, loc):
        nearestCluster = self.findNearestCluster(loc)
        dist = loc.getDistance(nearestCluster.getCentroid())
        # check whether tweet fits into closest cluster
        if dist > self.computeMBS(nearestCluster):
            self._noHitCount += 1
            return -1
        else:
            return nearestCluster.getId()

    # Find the closest cluster
    def findNearestCluster(self, loc):
        minDist = sys.float_info.max
        # get the closest cluster
        nearestCluster = None
        for cluster in self._clusters.values():
            centroid = cluster.getCentroid()
            dist = loc.getDistance(centroid)
            if dist < minDist:
                nearestCluster = MicroCluster.MicroCluster_from_other(cluster)
                minDist = dist
        return nearestCluster

    def computeMBS(self, cluster):
        size = cluster.size()
        centroid = cluster.getCentroid()
        if(size > 1):
            # when there are multiple points, calc the RMS as the boundary distance
            squareSum = cluster.getSquareSum().getL1Norm()
            centroidNorm = centroid.getNorm()
            boundaryDistance = math.sqrt(abs(squareSum/size - centroidNorm*centroidNorm))
        else:
            #if there is one point in the cluster, find the distance to the nearest neighbor
            boundaryDistance = sys.maxsize
            for neighbor in self._clusters.values():
                # do not count the cluster itself
                if neighbor.getId() == cluster.getId():
                    continue
                otherCentroid = neighbor.getCentroid()
                dist = otherCentroid.getDistance(centroid)
                if(dist < boundaryDistance):
                    boundaryDistance = dist
        return boundaryDistance * self._mbsFactor

    # create a new cluster for a single point.
    def createNewCluster(self, tweet):
        list = []
        list.append(tweet)
        self._clusters[self._toAssignClusterId] = MicroCluster.MicroCluster_from_tweets_id(list, self._toAssignClusterId)
        self._toAssignClusterId += 1

    # delete outdated clusters
    def removeOutdated(self):
        removeIDs = []
        for cluster in self._clusters.values():
            # try to forget old cluster
            freshness = cluster.getFreshness(self._quantile)
            if(self._currentTimestamp - freshness > self._outdatedThreshold):
                removeIDs.append(cluster.getId())
        for id in removeIDs:
            del self._clusters[id]
        self._outdatedCount += len(removeIDs)

    #Merge existing clusters
    def mergeCluster(self):
        # the number of merge operations that need to be performed
        toMergeCnt = len(self._clusters) * (1 - self._mc)
        # create heap
        pairHeap = []
        # compute pair-wise similarities among current clusters and update the heap.
        self.updateHeap(pairHeap)
        # mergeMap is used to record the merge history: <original cluster id, new cluster id>
        mergeMap = dict()
        mergeCnt = 0
        while len(pairHeap) != 0:
            if mergeCnt >= toMergeCnt:
                break
            pair = heapq.heappop(pairHeap)
            idA = pair._clusterA.getId()
            idB = pair._clusterB.getId()
            mergedA = idA in mergeMap
            mergedB = idB in mergeMap
            if not mergedA and not mergedB:
                # when neither A and B have been merged before, merge B into A, and delete B from the current cluster set.
                pair._clusterA.merge(pair._clusterB)
                mergeMap[idA] = idA
                mergeMap[idB] = idA
                del self._clusters[idB]
            elif mergedA and not mergedB:
                # when A has been merged before and B has not, merge B into A, and delete B from the list.
                bigClusterId = mergeMap[idA]
                bigCluster = self._clusters[bigClusterId]
                bigCluster.merge(pair._clusterB)
                mergeMap[idB] = bigClusterId
                del self._clusters[idB]
            elif not mergedA and mergedB:
                # when B has been merged and A has not, merge A into B, and delete A from the list.
                bigClusterId = mergeMap[idB]
                bigCluster = self._clusters[bigClusterId]
                bigCluster.merge(pair._clusterA)
                mergeMap[idA] = bigClusterId
                del self._clusters[idA]
            else:
                # when A and B have both been merged, check whether they belong to the same composite cluster, if yes, then no action
                # otherwise, merge bigB to bigA.
                bigClusterIdA = mergeMap[idA]
                bigClusterIdB = mergeMap[idB]
                if bigClusterIdA != bigClusterIdB:
                    bigClusterA = self._clusters[bigClusterIdA]
                    bigClusterA.merge(self._clusters[bigClusterIdB])
                    # update the member clusters of bidClusterB in merge map
                    needUpdate = set()
                    for k,v in mergeMap.items():
                        if v == bigClusterIdB:
                            needUpdate.add(k)
                    for updateId in needUpdate:
                        mergeMap[updateId] = bigClusterIdA
                    del self._clusters[bigClusterIdB]
            mergeCnt += 1



    def updateHeap(self, pairHeap):
        list_clusters = list(self._clusters.values())
        for i in range(0, len(list_clusters)):
            centroidA = list_clusters[i].getCentroid()
            for j in range(i+1, len(list_clusters)):
                centroidB = list_clusters[j].getCentroid()
                dist = centroidA.getDistance(centroidB)
                heapq.heappush(pairHeap, MicroClusterPair(list_clusters[i], list_clusters[j], dist))

    # #/***************************  Pyramid Time Frame ****************************/
    def getPyramid(self):
        return self._ptf

    def updatePyramid(self):
        if(self._ptf.isNewTimeFrame(self._currentTimestamp)):
            self._ptf.storeSnapshot(self._currentTimestamp, self._clusters)


    # /***************************  I/O and utils ****************************/
    # get the most updated clusters
    def getRealTimeClusters(self):
        return set(self._clusters.values())

    # def writeClusters(self):
    #     pass
    #
    def printStats(self):
        s = "Clustream Stats:"
        s += " current timestamp: " + str(self._currentTimestamp)
        s += "; # processed tweets=" + str(self._numOfProcessedTweets)
        s += "; elapsedTime=" + str(self._elapsedTime)
        s += "; current # cluster=" + str(len(self._clusters))
        s += "; noHitCount=" + str(self._noHitCount)
        s += "; outdatedCount=" + str(self._outdatedCount)
        s += "; mergeNumber=" + str(self._mergeNumber)
        s += "; pyramid size=" + str(self._ptf.size())
        print(s)

    def getNumOfProcessedTweets(self):
        return self._numOfProcessedTweets

    def getElapsedTime(self):
        return self._elapsedTime



