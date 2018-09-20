import math
from clustream.snapshot import Snapshot
from collections import deque
class Pyramid(object):

    def __init__(self):
        self._orderMap = dict()
        self._snapshots = dict()
        self._currentTimeFrameId = -1
        self._ll = 3
        self._alpha = 2
        self._timeFrameGranularity = 3600

        self._capacity = int(math.pow(self._alpha, self._ll)) + 1

    @staticmethod
    def Pyramid_with_Params(alpha, ll, timeFrameGranularity):
        pyramid = Pyramid()
        pyramid._alpha = alpha
        pyramid._ll = ll
        pyramid._capacity = int(math.pow(pyramid._alpha, pyramid._ll)) + 1
        pyramid._timeFrameGranularity = timeFrameGranularity
        return pyramid


    def size(self):
        return len(self._snapshots)

    def isNewTimeFrame(self, timestamp):
        return (timestamp/self._timeFrameGranularity) != self._currentTimeFrameId

    def toTimeFrameId(self, timestamp):
        return int(timestamp/self._timeFrameGranularity)

    def storeSnapshot(self, timestamp, clusters):
        timeFrameId = self.toTimeFrameId(timestamp)
        self._currentTimeFrameId = timeFrameId
        # compute which order the snapshot should go to
        order = self.getOrder(timeFrameId)
        # store the snapshot into the corresponding order
        if order in self._orderMap:
            #appending to the list, the last one in the list is the newest snapshot
            llist = deque([])
            llist = self._orderMap[order]
            # If the time frame id is already contained in the pyramid, do nothing
            for existingId in llist:
                if timeFrameId == existingId:
                    return
            llist.append(timeFrameId)
            if len(llist) > self._capacity:
                # when the layer is full, delete the oldest snapshot and free the memory
                removed = llist.popleft()
                del self._snapshots[removed]
        else:
            llist = deque([])
            llist.append(timeFrameId)
            self._orderMap[order] = llist
        #write the snapshot (in memory)
        snapshot = Snapshot(order, timeFrameId, timestamp, clusters)
        self._snapshots[timeFrameId] = snapshot

    # find the deepest order for the timeframeid
    def getOrder(self, timeFrameId):
        if (timeFrameId == 0):
            return 0
        order = 0
        tmp = self._alpha
        while (timeFrameId % tmp == 0):
            order += 1
            tmp = int(math.pow(self._alpha, order + 1))
        return order



    # find the deepest order for the timeframeid
    def getOder(self, timeFrameId):
        if timeFrameId == 0:
            return 0
        order = 0
        tmp = self.alpha
        while timeFrameId % tmp == 0:
            order += 1
            tmp = int(math.pow(self.alpha, order + 1))
        return order

    #/************************** Functions for retrieving snapshot(s) **************************/
    def loadSnapshot(self, timeFrameId):
        return self._snapshots[timeFrameId]

    def getSnapshotJustBefore(self, timestamp):
        queryTimeFrameId = self.toTimeFrameId(timestamp)
        timeFrameId = self.findTimeFrameJustBefore(queryTimeFrameId)
        return self.loadSnapshot(timeFrameId)

    def getSnapshotsBetween(self, startTimestamp, endTimestamp):
        startTimeFrameId = self.toTimeFrameId(startTimestamp)
        endTimeFrameId = self.toTimeFrameId(endTimestamp)
        timeFrameIds = self.findTimeFrameBetween(startTimeFrameId, endTimeFrameId)
        results = []
        for timeFrameId in timeFrameIds:
            snapshot = self.loadSnapshot(timeFrameId)
            if snapshot == None:
                print("Null snapshot! %d"%timeFrameId)
            results.append(snapshot)
        return results

    # find the snapshot for a given timestamp, if there is no exact match, return the nearest
    def findTimeFrameJustBefore(self, queryTimeFrameId):
        order = self.getOrder(queryTimeFrameId)
        # note that the snapshots are stored from old to new, namely ascending order of the queryTimeFrameId
        list = self._orderMap[order]
        # if this is an exact match, return the snapshot
        if queryTimeFrameId in list:
            return queryTimeFrameId
        # if no exact match, then find the nearest snapshot before this query queryTimeFrameId
        mostRecent = -1
        for sameOrderList in self._orderMap.values():
            for i in range(0, len(sameOrderList)):
                ts = sameOrderList[i]
                if (ts > mostRecent):
                    mostRecent = ts
                if (ts > queryTimeFrameId):
                    break
        # find the nearest snapshot.
        if (mostRecent == -1):
            for sameOrderList in self._orderMap.values():
                ts = sameOrderList[0]
                if mostRecent == -1 or ts < mostRecent:
                    mostRecent = ts
        return mostRecent


    # find the sorted timeFrameIds that fall in a range.
    def findTimeFrameBetween(self, startTimeFrameId, endTimeFrameId):
        results = []
        for sameOrderList in self._orderMap.values():
            for i in range(0, len(sameOrderList)):
                ts = sameOrderList[i]
                if ts > endTimeFrameId:
                    break
                if ts >= startTimeFrameId and ts <= endTimeFrameId:
                    results.append(ts)
        results.sort()
        return results

    def printStats(self):
        ret = "Order map: "
        for sameOrderList in self._orderMap.values():
            ret += str(sameOrderList) + "\n"
        ret += "Snapshots: " + str(self._snapshots.keys())
        return ret