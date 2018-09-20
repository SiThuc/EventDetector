import time
from utils.utils import Utils
from utils.utils import mean
from utils.scorecell import ScoreCell
from rank.referenceperiod import ReferencePeriod

class Ranker(object):
    def __init__(self, clustream, weighter, eta):
        self._clustream = clustream
        self._weighter = weighter
        self._eta = eta

    def rank(self, clusters, bandwidth, startTimestamp, endTimestamp, refTimespan):
        start = time.time()
        # init the reference periods
        self.initReferencePeriods(startTimestamp, refTimespan)
        # init the probability distribution for each cluster
        for gec in clusters:
            gec.genProbDistribution()
        end = time.time()
        self._timeFetching = end - start
        self._numClusters = len(clusters)
        self._numReferencePeriods = len(self._referencePeriods)

        start = time.time()
        scorecells = []
        for clusterIndex in range(0, len(clusters)):
            cluster = clusters[clusterIndex]
            score = self.calcScore(cluster, clusters, bandwidth, startTimestamp, endTimestamp)
            cluster.setScore(score)
            sc = ScoreCell(clusterIndex, score)
            scorecells.append(sc)
        self.sortScoreCells(scorecells)
        end = time.time()
        self._timeRanking = end - start
        return scorecells

    # fetch from clustream the list of reference periods and the clusters falling inside each reference period
    def initReferencePeriods(self, startTimestamp,refTimespan):
        self._referencePeriods = []
        # get the snapshots falling in the reference window, ordered from old to new.
        snapshots = self._clustream.getPyramid().getSnapshotsBetween(startTimestamp - refTimespan, startTimestamp)
        for i in range(0, len(snapshots)-1):
            startSnapshot = snapshots[i]
            endSnapshot = snapshots[i+1]
            rp = ReferencePeriod(startSnapshot, endSnapshot)
            self._referencePeriods.append(rp)

    # get the weighted zscore as the final score for the cluster
    def calcScore(self, cluster, allCandidates, bandwidth, startTimestamp, endTimestamp):
        temporalZScores = self.genTemporalZScores(cluster, bandwidth, startTimestamp, endTimestamp); # key: entity ID, value: probability
        spatialZScores = self.genSpatialZScores(cluster, allCandidates); # key: entity ID, value: probability
        self._weighter.buildTFIDF(cluster)
        weights = self._weighter.getWeights()
        score = 0
        for entity in temporalZScores.keys():
            temporalZscore = temporalZScores[entity]
            spatialZscore = spatialZScores[entity]
            weight = weights[entity]
            score += weight * (self._eta * temporalZscore + (1 - self._eta) * spatialZscore)
        return score


#   /*****************************************temporal z-score ********************************/
    # get the z-score vector for the entities in the current cluster, key: entity id, value: z-score
    def genTemporalZScores(self, cluster, bandwidth, startTimestamp, endTimestamp):
        # the count of occurrences in the online cluster. Key: entity Id, value: occurrence.
        onlineOccurrences = cluster.getEntityOccurrences()
        referenceOccurrencesList = self.getReferenceOccurrencesList(cluster, bandwidth, startTimestamp, endTimestamp)
        return self.calcTemporalZScore(onlineOccurrences, referenceOccurrencesList)

    # get the list of reference entity occurrences in the reference time window.
    def getReferenceOccurrencesList(self, cluster, bandwidth, startTimestamp, endTimestamp):
        referenceOccurrencesList = []
        # the set of entity ids in the online cluster
        entities = cluster.getEntities()
        # get the snapshots falling in the reference window, ordered from old to new.
        for rp in self._referencePeriods:
            referenceOccurrences = rp.getExpectedOccurrences(entities, cluster.getCenter().getLocation(), bandwidth, endTimestamp - startTimestamp)
            referenceOccurrencesList.append(referenceOccurrences)
        return referenceOccurrencesList

    def calcTemporalZScore(self, onlineOccurrences, references):
        zscores = dict()
        for k,v in onlineOccurrences.items():
            entity = k
            cnt = v
            # extract the reference numbers
            referenceCounts = []
            for referenceOccurrences in references:
                referenceCounts.append(referenceOccurrences[entity])

            if len(referenceCounts) == 0:
                print("No reference count when computing temporal z-score!")
                zscores[entity] = 0
            else:
                # calc z-score for the specific entity
                mu = mean(referenceCounts)
                sigma = Utils.std(referenceCounts)
                if sigma == 0:
                    zscore = 0.0
                else:
                    zscore = (cnt - mu) / sigma
                zscores[entity] = zscore
        return zscores


#     /***************************************** spatial z-score ********************************/
    # get the spatial z-score vector for the entities in the current cluster, key: entity id, value: spatial z-score
    def genSpatialZScores(self, cluster, allCandidates):
        zscores = dict()
        for entity in cluster.getEntities():
            prob = cluster.getEntityProb(entity)
            # extract the reference probabilities
            refProbs = []
            for g in allCandidates:
                refProbs.append(g.getEntityProb(entity))
            if len(refProbs) == 0:
                print("No reference count when computing spatial z-score!")
                zscores[entity] = 0
            else:
                # calc z-score for the specific entity
                mu = mean(refProbs)
                sigma = Utils.std(refProbs)

                if sigma == 0:
                    zscore = 0
                else:
                    zscore = (prob - mu) / sigma

                zscores[entity] = zscore
        return zscores

    # rank the scorecells in the descending order of the score
    def sortScoreCells(self, scoreCells):
        scoreCells.sort()

    def printStats(self):
        s = "Ranker Stats:"
        s += " numClusters=" + str(self._numClusters)
        s += "; numReferencePeriods=" + str(self._numReferencePeriods)
        s += "; timeFetchingClustream=" + str(self._timeFetching)
        s += "; timeRanking=" + str(self._timeRanking)
        print(s)

    def getNumReferencePeriods(self):
        return self._numReferencePeriods

    def getTimeFetching(self):
        return self._timeFetching

    def getTimeRanking(self):
        return self._timeRanking

































