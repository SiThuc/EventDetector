import copy
from clustream.microcluster import MicroCluster
class Snapshot(object):
    def __init__(self, order, timeFrameId, timestamp, clusters):
        self._order = order
        self._timeFrameId = timeFrameId
        self._timestamp = timestamp
        self._clusters = dict()
        for key, value in clusters.items():
            clusterId = int(key)
            cluster = value
            self._clusters[clusterId] = cluster

    def getClusters(self):
        return self._clusters

    def getTimestamp(self):
        return self._timestamp

# get the different clusters by subtracting a previous snapshot
    def getDiffClusters(self, prevSnapshot):
        beforeMap = prevSnapshot.getClusters()
        endMap = self._clusters
        diffSet = set()
        for originalCluster in endMap.values():
            base = MicroCluster.MicroCluster_from_other(originalCluster)
            if base.isSingle():
                if base._clusterId in beforeMap:
                    before = beforeMap[base._clusterId]
                    base.subtract(before)
            else:
                clusterIDSet = base._idSet
                clusterIDSet.add(base._clusterId)
                for cid in clusterIDSet:
                    if cid in beforeMap:
                        before = beforeMap[cid]
                        base.subtract(before)
            if base._num > 0:
                diffSet.add(base)
        return diffSet

    def __str__(self):
        sb = ""
        sb += str(self._timestamp)
        sb += "=" + str(self._order)
        for cluster in self._clusters.values():
            sb += "=" + cluster.__str__()
        return sb


















