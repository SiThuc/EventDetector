
class TweetCluster(object):
    def __init__(self, center = None):
        self._center = center
        self._members = []
        self._authority = []
        self._distribution = dict()
        self._score = 0
        self._tfIdf = dict()

    def add(self, e, score):
        self._members.append(e)
        self._authority.append(score)

    def getScore(self):
        return self._score

    def getId(self):
        return self._clusterId

    def getCenter(self):
        return self._center

    def size(self):
        return len(self._members)

    def getEntities(self):
        return self._distribution.keys()

    def getEntityProb(self, entity):
        if entity in self._distribution:
            return self._distribution[entity]
        else:
            return 0

    def getMembers(self):
        return self._members

    # generate the probability distribution of entities
    def genProbDistribution(self):
        entityOccurrence = self.getEntityOccurrences()
        totalWeight = 0
        for weight in entityOccurrence.values():
            totalWeight += weight
            for entity in entityOccurrence.keys():
                probability = entityOccurrence[entity] / totalWeight
                self._distribution[entity] = probability

    def getEntityOccurrences(self):
        occurrences = dict()
        for e in self._members:
            entities = e.getEntities()
            for entity in entities:
                if entity in occurrences:
                    originalCnt = occurrences[entity]
                else:
                    originalCnt = 0
                occurrences[entity] = originalCnt+ 1

        return occurrences

    def setScore(self, score):
        self._score = score

    def setTfIdf(self, entity, val):
        self._tfIdf[entity] = val

    def __str__(self):
        s = "# Cluster Score:" + str(self._score) + "\n"
        s += "Num of Tweets:" + str(len(self._members)) + "\n"
        s += "Center Tweet ID:" + str(self._center.getTweetId()) + "\n"
        #for e in self._members:
        #    s += e.__str__() + "\n"
        return s

    def toJson(self):
        tweets = []
        i = 1
        for tw in self._members:
            vl = dict()
            lng = tw.getLocation().getLng()
            lat = tw.getLocation().getLat()
            text = tw.getEntities()
            vl['num'] = i
            vl['lng'] = lng
            vl['lat'] = lat
            vl['text'] = text
            tweets.append(vl)
            i += 1

        data = {
            "Score":self._score,
            "NumberOfTweet": len(self._members),
            "CenterTweetId":self._center.getTweetId(),
            "Members": tweets
        }
        return data
