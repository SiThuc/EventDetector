
class Query(object):
    def __init__(self, start, end, refWindowSize, minSup):
        self._startTS = start
        self._endTS = end
        if self._startTS - refWindowSize < 0:
            self._startRefTS = 0
        else:
            self._startRefTS = self._startTS - refWindowSize
        self._endRefTS = self._startTS
        self._minSup = minSup


    def getStartTS(self):
        return self._startTS

    def getEndTS(self):
        return self._endTS

    def getRefStartTS(self):
        return self._startRefTS

    def getRefEndTS(self):
        return self._endRefTS

    def getMinSup(self):
        return self._minSup

    def getQueryInterval(self):
        self._endTS - self._startTS

    def printQuery(self):
        print("StartTS:" + str(self._startTS) + ",")
        print("endTS:" + str(self._endTS) + ",")
        print("StartRefTS:" + str(self._startRefTS) + ",")
        print("EndRefTS:" + str(self._endRefTS) + ",")
        print("Minsup:" + str(self._minSup))
