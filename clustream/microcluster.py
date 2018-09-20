import copy
import math
from geo.ArrayRealVector import ArrayRealVector

class MicroCluster(object):
    MU_THRESHOLD = 50

    def __init__(self, id = -1, ts1 = 0, ts2 = 0, num = 0, sum=ArrayRealVector(), ssum=ArrayRealVector()):
        self._clusterId = id
        self._sum = sum
        self._ssum = ssum
        self._ts1 = ts1
        self._ts2 = ts2
        self._num = num
        self._idSet = None
        self._words = None

    @staticmethod
    def MicroCluster_from_other(other):
        micro = copy.deepcopy(other)
        return micro

    # Initialize a cluster with the given list of tweets and cluster id.
    @staticmethod
    def MicroCluster_from_tweets_id(memberList, id):
        micro = MicroCluster()
        micro._clusterId = id
        micro._num = 0
        micro._sum = ArrayRealVector.ArrayRealVector_with_Dim(2)
        micro._ssum = ArrayRealVector.ArrayRealVector_with_Dim(2)
        micro._words = dict()
        for tweet in memberList:
            micro.absorb(tweet)
        return micro

    # whether this is a single cluster or not.
    def isSingle(self):
        return self._idSet == None

    def size(self):
        return self._num

    def getId(self):
        return self._clusterId

    def getSum(self):
        return self._sum

    def getSquareSum(self):
        return self._ssum

    def getCentroid(self):
        return self._sum.mapDivide(self._num)

    def getWords(self):
        return self._words

    def getFreshness(self, quantile):
        muTime = float(self._ts1 / self._num)
        # If there are too few tweets
        if self._num < MicroCluster.MU_THRESHOLD:
            return muTime
        sigmaTime = math.sqrt(self._ts2/self._num - math.pow((self._ts1/self._num),2))
        return muTime + sigmaTime*quantile

    def absorb(self, tweet):
        self._num += 1
        self._ts1 += tweet.getTimestamp()
        self._ts2 += math.pow(tweet.getTimestamp(), 2)
        loc = tweet.getLocation().toRealVector()
        self._sum = self._sum.add(loc)
        self._ssum = self._ssum.add(loc.ebeMultiply(loc))
        for word in tweet.getEntities():
            cnt = 0
            if word in self._words:
                cnt = self._words[word]
            cnt += 1
            self._words[word] = cnt

    # merge other to this cluster
    def merge(self, other):
        # 1. update iList
        if self._idSet == None:
            self._idSet = set()
        self._idSet.add(other._clusterId)
        # If other is a composite cluster, then we need to incorporate the member ids.
        if other._idSet != None:
            self._idSet.union(other._idSet)

        # 2. update num, ts1, ts2, sum, ssum
        self._num += other._num
        self._ts1 += other._ts1
        self._ts2 += other._ts2
        self._sum = self._sum.add(other.getSum())
        self._ssum = self._ssum.add(other.getSquareSum())

        #3. update the semantic information
        for word, cnt in other.getWords().items():
            originalCnt = 0
            if word in self._words:
                originalCnt = self._words[word]
            self._words[word] = originalCnt + cnt

    def subtract(self, other):
        self._sum = self._sum.subtract(other.getSum())
        self._ssum = self._ssum.subtract(other.getSquareSum())
        self._num -= other._num
        self._ts1 -= other._ts1
        self._ts2 -= other._ts2

        for word in list(other._words.keys()):
            originalCnt = 0
            if word in self._words:
                originalCnt = self._words[word]

            cnt = other._words[word]

            if originalCnt == cnt:
                del self._words[word]
            else:
                self._words[word] = originalCnt - cnt

            # if originalCnt < cnt:
            #     self._words[word] = cnt - originalCnt
            # elif originalCnt == cnt:
            #     del self._words[word]
            # else:
            #     self._words[word] = originalCnt - cnt

        # for k,v in other.getWords().items():
        #     word = k
        #     cnt = v
        #     originalCnt = 0
        #     if word in self._words:
        #         originalCnt = self._words[word]
        #
        #     if(originalCnt < cnt):
        #         print("Original:",originalCnt)
        #         print("cnt:", cnt)
        #         print("Original count is smaller than the new count!")
        #         exit(0)
        #     elif originalCnt == cnt:
        #         del self._words[word]
        #     else:
        #         self._words[word] = originalCnt - cnt

    def __str__(self):
        itemSep = "+"
        sb = ""
        sb += str(self._clusterId) + itemSep
        if self._idSet == None:
            sb += 0 + itemSep
        else:
            sb += 1 + itemSep
        sb += str(self._num) + itemSep
        sb += str(self._ts1) + itemSep
        sb += str(self._ts2) + itemSep
        sb += str(self._sum) + itemSep
        sb += str(self._ssum) + itemSep
        if self._idSet != None:
            for id in self._idSet:
                sb += str(id) + " "
        sb += itemSep
        sb += self._words
        return sb