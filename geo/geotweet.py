import sys
import re
import location
# *
# * Created by Si Thuc
#
from geo.location import Location


class GeoTweet(object):
    def __init__(self, tweetId, timestamp, loc, entities):
        self._tweetId = tweetId
        self._timestamp = timestamp
        self._loc = loc
        self._entities = entities

    def geoTweetFromAString(tweetString):
        temp = re.split('\\t', tweetString)
        tweetId = int(temp[0])
        timestamp = int(temp[1])
        lng = float(temp[2])
        lat = float(temp[3])
        loc = Location(lng, lat)
        entities = temp[4:]
        tweet = GeoTweet(tweetId, timestamp, loc, entities)
        return tweet

    def getTweetId(self):
        return self._tweetId

    def numEntity(self):
        return len(self._entities)

    # def getUserId(self):
    #     return self._userId

    # calculate the weighted spatio-semantic score received from another geo-tweet
    def calcScoreFrom(self, e, bandwidth, entityGraph):
        geoScore = self.calcKernelScore(e, bandwidth)
        semanticScore = e.calcGraphDistFrom(entityGraph, e)
        return geoScore * semanticScore

    # calc the geographical kernel score between two entities
    def calcKernelScore(self, other, bandwidth):
        dist = self.calcGeoDist(other)
        kernelScore = 1.0 - (dist / bandwidth) * (dist / bandwidth)  # kernel
        return kernelScore

    # calc the Euclidean distance between two geo tweets.
    def calcGeoDist(self, other):
        return self._loc.calcEuclideanDist(other.getLocation())

    def calcGraphDistFrom(self, graph, other):
        proximity = 0.0
        for entity in self.getEntities():
            for otherEntitity in other.getEntities():
                tempRWR = graph.getRWR(otherEntitity, entity)
                proximity += tempRWR
        size = len(self.getEntities()) * len(other.getEntities())
        return proximity/size

    def getTimestamp(self):
        return self._timestamp

    def getLocation(self):
        return self._loc

    def getEntities(self):
        return self._entities

    def __str__(self):
        self._loc.__str__() + ", " + self._entities.__str__()

    # def toString(self):
    #     return self.loc.toString + " ," + self.entityIds

























