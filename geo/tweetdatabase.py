import sys
import copy
from geo.geotweet import GeoTweet

#This class keeps the tweets in a specific time span.
class TweetDatabase(object):

    def __init__(self):
        self._tweets = []
        self._startTimestamp = sys.maxsize
        self._endTimestamp = -sys.maxsize -1

    def getStartTimestamp(self):
        return self._startTimestamp

    def getEndTimestamp(self):
        return self._endTimestamp

    def setNewStartTimestamp(self, newStart):
        self._startTimestamp = newStart

    def getTweets(self):
        return self._tweets

    #get one tweet by index
    def getTweet(self, index):
        return self._tweets[index]

    def load(self, tweetFile):
        with open(tweetFile) as f:
            lines = f.readlines()
        f.close()

        for line in lines:
            if line.isNull():
                break
            gt = GeoTweet(line)
            self._tweets.append(gt)

    # delete the first #num tweets in the database
    def deleteFromHead(self, num):
        if len(self._tweets) - num <= 0:
            return
        self._startTimestamp = self._tweets[num].getTimestamp()
        updatedTweets = []
        for i in range(num, len(self._tweets)):
            updatedTweets.append(self._tweets[i])
        self._tweets = copy.deepcopy(updatedTweets)

    #delete the first #num tweets in the database until the first tweet is newer than startTS.
    def deleteFromHeadByTime(self, startTS):
        updatedTweets = []
        deletedTweets = []
        for i in range(0,len(self._tweets)):
            t = self._tweets[i]
            if t.getTimestamp() > startTS:
                updatedTweets.append(t)
            else:
                deletedTweets.append(t)
        self._tweets = copy.deepcopy(updatedTweets)
        return deletedTweets

    # append the new tweets to the end
    def addAll(self, td):
        self._tweets.append(td.getTweets())
        self._endTimestamp = td.getEndTimestamp()

    def add(self, tweet):
        self._tweets.append(tweet)
        if tweet.getTimestamp() < self._startTimestamp:
            self._startTimestamp = tweet.getTimestamp()
        if tweet.getTimestamp() > self._endTimestamp:
            self._endTimestamp = tweet.getTimestamp()

    def getGeoTweetsMap(self):
        results = dict()
        for gt in self._tweets:
            results.update({gt.getTweetId(): gt})
        return results

    def size(self):
        return len(self._tweets)