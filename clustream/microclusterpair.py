class MicroClusterPair:
    def __init__(self, cluster1, cluster2, dist):
        self._clusterA = cluster1
        self._clusterB = cluster2
        self._dist = dist

    def getDist(self):
        return self._dist

    def __lt__(self, other):
        return self._dist < other._dist